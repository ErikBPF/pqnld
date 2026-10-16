//! bias_probe — measure whether the label bound to an option moves pqnld's
//! decision distribution, and record an engine-validation arm.
//!
//! It talks to a running pqnld sidecar over HTTP only, so it never needs engine
//! internals: the sidecar's `label_seed` question field selects the label
//! binding for one question. Two factors are sampled per question:
//!
//! * **B (label):** option order fixed, labels shifted by seed `s` for `s` in
//!   `0..k`. Each label occupies each position exactly once across the run.
//! * **A (position):** option order rotated by `r` while `label_seed = r` keeps
//!   each option's label fixed, so only its position moves.
//!
//! Repeats of one request measure the within-session floor. The floor that
//! bounds a verdict is the cross-run one, computed by `--summarize` over arm
//! directories from separate engine runs.
//!
//! Modes:
//!   bias_probe --self-test
//!   bias_probe --capability-probe --base-url URL --out DIR
//!   bias_probe --fixture FILE --arm NAME --out DIR --base-url URL [--repeats 3] [--assert]
//!   bias_probe --summarize DIR [DIR ...]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

const REJECT_SPREAD: f64 = 0.02;
const REJECT_FLIPS: f64 = 0.02;
const SUPPORT_SPREAD: f64 = 0.01;
const SUPPORT_FLIPS: f64 = 0.005;
const NOISE_LIMIT: f64 = 0.01;

/// One scored permutation of a question.
#[derive(Clone, Debug)]
struct Sample {
    /// Label index carried by each option index.
    labels: Vec<usize>,
    /// Prompt position of each option index.
    positions: Vec<usize>,
    /// Softmaxed probability per option index, in fixture option order.
    probs: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct QuestionResult {
    id: String,
    keys: Vec<String>,
    expected: Option<String>,
    k: usize,
    /// Factor B: labels move, order fixed.
    label_samples: Vec<Sample>,
    /// Factor A: positions move, labels fixed per option.
    position_samples: Vec<Sample>,
    /// Identical repeats of one request.
    repeats: Vec<Vec<f64>>,
    refusals: Vec<String>,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx]
}

/// Per-option spread (max - min) across samples.
fn spreads(samples: &[Sample]) -> Vec<f64> {
    let k = samples.first().map(|s| s.probs.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(k);
    for option in 0..k {
        let values: Vec<f64> = samples.iter().map(|s| s.probs[option]).collect();
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        out.push(max - min);
    }
    out
}

fn argmax(probs: &[f64]) -> usize {
    let mut best = 0usize;
    for (i, p) in probs.iter().enumerate() {
        if *p > probs[best] {
            best = i;
        }
    }
    best
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// Fraction of samples whose argmax differs from the modal option.
fn flip_rate(samples: &[Sample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let k = samples[0].probs.len();
    let modal = (0..k)
        .max_by(|a, b| {
            let ma = mean(&samples.iter().map(|s| s.probs[*a]).collect::<Vec<_>>());
            let mb = mean(&samples.iter().map(|s| s.probs[*b]).collect::<Vec<_>>());
            ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let flips = samples.iter().filter(|s| argmax(&s.probs) != modal).count();
    flips as f64 / samples.len() as f64
}

/// Mean per-slot offset: for every (sample, option) pair, the option's
/// probability minus that option's mean probability, averaged per slot value.
/// `slot` selects the label or the position of an option in a sample.
fn slot_offsets(samples: &[Sample], slot: fn(&Sample, usize) -> usize, slots: usize) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let k = samples[0].probs.len();
    let means: Vec<f64> = (0..k)
        .map(|o| mean(&samples.iter().map(|s| s.probs[o]).collect::<Vec<_>>()))
        .collect();
    let mut sums = vec![0.0f64; slots];
    let mut counts = vec![0usize; slots];
    for sample in samples {
        for option in 0..k {
            let s = slot(sample, option);
            sums[s] += sample.probs[option] - means[option];
            counts[s] += 1;
        }
    }
    sums.iter()
        .zip(counts.iter())
        .map(|(s, c)| if *c == 0 { 0.0 } else { s / *c as f64 })
        .collect()
}

fn label_slot(sample: &Sample, option: usize) -> usize {
    sample.labels[option]
}

fn position_slot(sample: &Sample, option: usize) -> usize {
    sample.positions[option]
}

/// Pre-registered verdict. `noise` is the cross-run floor, `inversions` the
/// fraction of samples whose argmax ignored the fixture's expected answer.
fn verdict(spread_p95: f64, flips: f64, inversions: f64, noise: f64) -> &'static str {
    if noise > NOISE_LIMIT {
        return "noise-limited";
    }
    if spread_p95 > REJECT_SPREAD && flips > REJECT_FLIPS {
        "rejected"
    } else if spread_p95 <= SUPPORT_SPREAD && flips <= SUPPORT_FLIPS && inversions == 0.0 {
        "supported"
    } else {
        "unresolved"
    }
}

/// Aggregate one question into the reported metrics.
fn question_metrics(q: &QuestionResult, noise: f64) -> Value {
    let mut spread: Vec<f64> = spreads(&q.label_samples);
    let pos_spread = spreads(&q.position_samples);
    spread.extend(pos_spread.iter().cloned());
    let mut sorted = spread.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let flips_label = flip_rate(&q.label_samples);
    let flips_pos = flip_rate(&q.position_samples);
    let flips = flips_label.max(flips_pos);

    let inversions = q
        .expected
        .as_ref()
        .and_then(|expected| q.keys.iter().position(|k| k == expected))
        .map(|expected_option| {
            let mut samples: Vec<&Sample> = q.label_samples.iter().collect();
            samples.extend(q.position_samples.iter());
            if samples.is_empty() {
                0.0
            } else {
                samples.iter().filter(|s| argmax(&s.probs) != expected_option).count() as f64
                    / samples.len() as f64
            }
        })
        .unwrap_or(0.0);

    let floor = if q.repeats.len() > 1 {
        let k = q.repeats[0].len();
        let mut worst = 0.0f64;
        for option in 0..k {
            let values: Vec<f64> = q.repeats.iter().map(|r| r[option]).collect();
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            worst = worst.max(max - min);
        }
        worst
    } else {
        0.0
    };

    let p95 = percentile(&sorted, 0.95);
    json!({
        "id": q.id,
        "keys": q.keys,
        "expected": q.expected,
        "options": q.k,
        "label_samples": q.label_samples.len(),
        "position_samples": q.position_samples.len(),
        "refusals": q.refusals,
        "spread_p50": percentile(&sorted, 0.50),
        "spread_p95": p95,
        "spread_max": sorted.last().copied().unwrap_or(0.0),
        "flip_rate": flips,
        "flip_rate_label": flips_label,
        "flip_rate_position": flips_pos,
        "inversions": inversions,
        "repeat_floor": floor,
        "delta_label": slot_offsets(&q.label_samples, label_slot, q.k),
        "delta_position": slot_offsets(&q.position_samples, position_slot, q.k),
        "verdict": verdict(p95, flips, inversions, floor.max(noise)),
    })
}

impl QuestionResult {
    fn all_refusals(&self) -> bool {
        self.label_samples.is_empty() && self.position_samples.is_empty()
    }
}

fn sha256_of(path: &Path) -> String {
    use std::process::Command;
    Command::new("sha256sum")
        .arg(path)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.split_whitespace().next().map(|h| h.to_string()))
        .unwrap_or_default()
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
}

fn flag_present(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

struct Client {
    http: reqwest::Client,
    base: String,
}

impl Client {
    fn new(base: &str, timeout: u64) -> Client {
        Client {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(timeout))
                .build()
                .expect("http client"),
            base: base.trim_end_matches('/').to_string(),
        }
    }

    async fn post(&self, path: &str, body: &Value) -> Result<(u16, Value), String> {
        let response = self
            .http
            .post(format!("{}{path}", self.base))
            .json(body)
            .send()
            .await
            .map_err(|e| format!("{path}: {e}"))?;
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Ok((status, parsed))
    }

    async fn decide(&self, body: &Value) -> Result<Vec<f64>, String> {
        let (status, parsed) = self.post("/v1/decide", body).await?;
        if status != 200 {
            return Err(format!("{status}: {}", parsed));
        }
        let answer = parsed
            .get("answers")
            .and_then(Value::as_object)
            .and_then(|a| a.values().next())
            .ok_or_else(|| format!("no answers in {parsed}"))?;
        let probs = answer
            .get("probabilities")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("no probabilities in {answer}"))?;
        Ok(probs.values().map(|v| v.as_f64().unwrap_or(0.0)).collect())
    }
}

fn rotated(k: usize, offset: usize) -> Vec<usize> {
    (0..k).map(|i| (i + offset) % k).collect()
}

async fn run_question(
    client: &Client,
    state: &Value,
    id: &str,
    question: &Value,
    expected: Option<String>,
    repeats: usize,
) -> QuestionResult {
    let keys: Vec<String> = question
        .get("criteria")
        .and_then(Value::as_object)
        .map(|c| c.keys().cloned().collect())
        .unwrap_or_default();
    let k = keys.len();
    let mut result = QuestionResult {
        id: id.to_string(),
        keys,
        expected,
        k,
        ..Default::default()
    };

    let call = |order: Vec<usize>, seed: u64, question: Value| {
        let client = client;
        let state = state.clone();
        async move {
            let criteria: Map<String, Value> = question
                .get("criteria")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let mut ordered = Map::new();
            for index in order {
                if let Some(key) = criteria.keys().nth(index) {
                    ordered.insert(key.clone(), criteria[key].clone());
                }
            }
            let mut q = question.clone();
            if let Some(object) = q.as_object_mut() {
                object.insert("criteria".to_string(), Value::Object(ordered));
                object.insert("label_seed".to_string(), json!(seed));
            }
            let mut questions = Map::new();
            questions.insert("q".to_string(), q);
            client.decide(&json!({"state": state, "questions": questions})).await
        }
    };

    // Factor B: labels shift, option order fixed.
    for seed in 0..k as u64 {
        match call(rotated(k, 0), seed, question.clone()).await {
            Ok(probs) => result.label_samples.push(Sample {
                labels: rotated(k, seed as usize),
                positions: rotated(k, 0),
                probs,
            }),
            Err(e) => result.refusals.push(format!("label seed {seed}: {e}")),
        }
    }
    // Factor A: order rotates while each option keeps its label.
    for offset in 0..k {
        match call(rotated(k, offset), offset as u64, question.clone()).await {
            Ok(probs) => {
                let mapping = rotated(k, offset);
                let mut reordered = vec![0.0; k];
                for (position, option) in mapping.iter().enumerate() {
                    reordered[*option] = probs[position];
                }
                result.position_samples.push(Sample {
                    labels: rotated(k, 0),
                    positions: mapping,
                    probs: reordered,
                });
            }
            Err(e) => result.refusals.push(format!("order offset {offset}: {e}")),
        }
    }
    // Identical repeats for the within-session floor.
    for _ in 0..repeats {
        match call(rotated(k, 0), 0, question.clone()).await {
            Ok(probs) => result.repeats.push(probs),
            Err(e) => result.refusals.push(format!("repeat: {e}")),
        }
    }
    result
}

async fn capability_probe(client: &Client, out: &Path) -> Result<Value, String> {
    let mut report = Map::new();
    let (status, tokenize) = client
        .post("/tokenize", &json!({"content": "b", "add_special": false}))
        .await?;
    report.insert("tokenize".into(), json!({"status": status, "body": tokenize}));
    let (status, completion) = client
        .post(
            "/completion",
            &json!({"prompt": "Options:\na) Paris\nb) Berlin\nReply with one letter.",
              "n_predict": 1, "temperature": 0, "n_probs": 8}),
        )
        .await?;
    report.insert("completion_n_probs".into(), json!({"status": status, "body": completion}));
    let (status, props) = client.post("/props", &json!({})).await?;
    report.insert("props".into(), json!({"status": status, "body": props}));
    let verdict = if report["tokenize"]["status"] == 200
        && report["completion_n_probs"]["status"] == 200
        && report["completion_n_probs"]["body"]["completion_probabilities"][0]["top_logprobs"].is_array()
    {
        "native-top-k-surface-present"
    } else {
        "gap"
    };
    report.insert("verdict".into(), json!(verdict));
    std::fs::write(out.join("capabilities.json"), serde_json::to_string_pretty(&Value::Object(report)).unwrap())
        .map_err(|e| e.to_string())?;
    Ok(json!({"verdict": verdict}))
}

fn self_test() -> Result<(), String> {
    let k = 4usize;
    let unbiased = |seed: usize| Sample {
        labels: rotated(k, seed),
        positions: rotated(k, 0),
        probs: vec![0.7, 0.1, 0.1, 0.1],
    };
    // Null: every permutation gives the same distribution.
    let null: Vec<Sample> = (0..k).map(unbiased).collect();
    let null_spread = spreads(&null);
    let null_metrics = question_metrics(
        &QuestionResult {
            id: "synthetic-null".into(),
            keys: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            expected: Some("a".into()),
            k,
            label_samples: null.clone(),
            position_samples: null.clone(),
            repeats: vec![vec![0.7, 0.1, 0.1, 0.1]; 3],
            refusals: vec![],
        },
        0.0,
    );
    if null_spread.iter().any(|s| *s != 0.0) {
        return Err(format!("null samples should not spread: {null_spread:?}"));
    }
    if null_metrics["verdict"] != json!("supported") {
        return Err(format!("null should be supported: {null_metrics}"));
    }

    // Planted: the option carrying label index 3 is boosted by 0.3.
    let planted: Vec<Sample> = (0..k)
        .map(|seed| {
            let labels = rotated(k, seed);
            let mut probs = vec![0.2, 0.2, 0.2, 0.4];
            for (option, label) in labels.iter().enumerate() {
                if *label == 3 {
                    probs = vec![0.05, 0.05, 0.05, 0.85];
                    probs.swap(option, 3);
                    break;
                }
            }
            Sample { labels, positions: rotated(k, 0), probs }
        })
        .collect();
    let planted_metrics = question_metrics(
        &QuestionResult {
            id: "synthetic-planted".into(),
            keys: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            expected: Some("a".into()),
            k,
            label_samples: planted.clone(),
            position_samples: planted.clone(),
            repeats: vec![planted[0].probs.clone(); 3],
            refusals: vec![],
        },
        0.0,
    );
    if planted_metrics["verdict"] != json!("rejected") {
        return Err(format!("planted label bias should be rejected: {planted_metrics}"));
    }
    let deltas: Vec<f64> = planted_metrics["delta_label"]
        .as_array()
        .ok_or("no delta_label")?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0))
        .collect();
    let worst = (0..k).max_by(|a, b| deltas[*a].abs().partial_cmp(&deltas[*b].abs()).unwrap()).unwrap();
    if worst != 3 {
        return Err(format!("reported biased label {worst}, expected 3: {deltas:?}"));
    }

    // A floor above the band must suppress the verdict.
    let noisy = question_metrics(
        &QuestionResult {
            id: "synthetic-noisy".into(),
            keys: vec!["a".into(), "b".into(), "c".into(), "d".into()],
            expected: Some("a".into()),
            k,
            label_samples: planted.clone(),
            position_samples: planted.clone(),
            repeats: vec![vec![0.0, 0.0, 0.0, 0.0], vec![0.5, 0.0, 0.0, 0.0]],
            refusals: vec![],
        },
        0.0,
    );
    if noisy["verdict"] != json!("noise-limited") {
        return Err(format!("a 0.5 floor should be noise-limited: {noisy}"));
    }
    Ok(())
}

fn build_report(arm: &str, fixture: &Path, metrics: Vec<Value>, refusals: usize) -> Value {
    let mut sorted: Vec<f64> = metrics
        .iter()
        .filter_map(|m| m["spread_p95"].as_f64())
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = percentile(&sorted, 0.95);
    let flips = metrics
        .iter()
        .filter_map(|m| m["flip_rate"].as_f64())
        .fold(0.0f64, f64::max);
    let inversions = metrics
        .iter()
        .filter_map(|m| m["inversions"].as_f64())
        .fold(0.0f64, f64::max);
    // `flip_rate` and `inversions` are the worst question; the mean fields are the
    // fixture-wide figures a report should quote, and `accuracy_mean` is their
    // complement (agreement with the fixture's expected answers).
    let mean = |field: &str| {
        let values: Vec<f64> = metrics.iter().filter_map(|m| m[field].as_f64()).collect();
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    json!({
        "schema": "pqnld-bias-probe-v1",
        "arm": arm,
        "completed": now(),
        "fixture": fixture.display().to_string(),
        "fixture_sha256": sha256_of(fixture),
        "questions": metrics.len(),
        "refusals": refusals,
        "spread_p95": p95,
        "flip_rate": flips,
        "flip_rate_max": flips,
        "flip_rate_mean": mean("flip_rate"),
        "inversions": inversions,
        "inversions_max": inversions,
        "inversions_mean": mean("inversions"),
        "accuracy_mean": 1.0 - mean("inversions"),
        "verdict_within_run": verdict(p95, flips, inversions, 0.0),
        "metrics": metrics,
    })
}

/// Worst within-run repeat floor across the supplied runs.
fn repeat_floor(repeats: &[f64]) -> f64 {
    repeats.iter().cloned().fold(0.0f64, f64::max)
}

/// Run-to-run variation of one deciding metric across the supplied runs. It is a
/// floor across those runs only; whether they were separated by an engine
/// restart is the caller's responsibility and must be stated with any verdict.
fn cross_floor(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    if flag_present(&args, "--self-test") {
        match self_test() {
            Ok(()) => println!("PQNLD_BIAS_CORE_OK"),
            Err(e) => {
                eprintln!("self-test failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if flag_present(&args, "--summarize") {
        let start = args.iter().position(|a| a == "--summarize").unwrap() + 1;
        let dirs: Vec<PathBuf> = args[start..]
            .iter()
            .filter(|a| !a.starts_with("--"))
            .map(PathBuf::from)
            .collect();
        if dirs.is_empty() {
            eprintln!("--summarize needs at least one run directory");
            std::process::exit(1);
        }
        // Per question, gather the metric from every supplied run.
        let mut per_question: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        let mut runs = 0usize;
        for dir in &dirs {
            let Ok(entries) = std::fs::read_dir(dir) else { continue };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with("arm-") || path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                let Ok(value) = serde_json::from_str::<Value>(&text) else { continue };
                runs += 1;
                for metric in value["metrics"].as_array().cloned().unwrap_or_default() {
                    per_question
                        .entry(metric["id"].as_str().unwrap_or("").to_string())
                        .or_default()
                        .push(metric);
                }
            }
        }
        if runs == 0 {
            eprintln!("no arm-*.json found in {}", dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "));
            std::process::exit(1);
        }
        println!("runs: {runs} (floor is across these runs; restart separation is the caller's responsibility)");
        println!("{:<28} {:>6} {:>9} {:>7} {:>6} {:>9}  verdict", "question", "k", "spread95", "flips", "inv", "floor");
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for (id, metrics) in &per_question {
            let spread = metrics.iter().filter_map(|m| m["spread_p95"].as_f64()).fold(0.0f64, f64::max);
            let flips = metrics.iter().filter_map(|m| m["flip_rate"].as_f64()).fold(0.0f64, f64::max);
            let inv = metrics.iter().filter_map(|m| m["inversions"].as_f64()).fold(0.0f64, f64::max);
            let repeats: Vec<f64> = metrics.iter().filter_map(|m| m["repeat_floor"].as_f64()).collect();
            let spreads: Vec<f64> = metrics.iter().filter_map(|m| m["spread_p95"].as_f64()).collect();
            let run_flips: Vec<f64> = metrics.iter().filter_map(|m| m["flip_rate"].as_f64()).collect();
            let floor = repeat_floor(&repeats)
                .max(cross_floor(&spreads))
                .max(cross_floor(&run_flips));
            let verdict = verdict(spread, flips, inv, floor);
            *counts.entry(verdict).or_default() += 1;
            println!(
                "{id:<28} {:>6} {spread:>9.4} {flips:>7.3} {inv:>6.3} {floor:>9.4}  {verdict}",
                metrics[0]["options"].as_u64().unwrap_or(0)
            );
        }
        let summary: Vec<String> = counts.iter().map(|(k, v)| format!("{k}={v}")).collect();
        println!("questions: {}  verdicts: {}", per_question.len(), summary.join(" "));
        println!("note: a label-specific effect requires delta_label to exceed delta_position; both are in the arm JSON");
        println!("PQNLD_VERDICT_OK");
        return;
    }

    let out = PathBuf::from(flag_value(&args, "--out").unwrap_or_else(|| ".".to_string()));
    if let Err(e) = std::fs::create_dir_all(&out) {
        eprintln!("cannot create {}: {e}", out.display());
        std::process::exit(1);
    }
    let base = flag_value(&args, "--base-url").unwrap_or_else(|| "http://127.0.0.1:11560".to_string());
    let timeout: u64 = flag_value(&args, "--timeout").and_then(|t| t.parse().ok()).unwrap_or(600);
    let client = Client::new(&base, timeout);

    if flag_present(&args, "--capability-probe") {
        match capability_probe(&client, &out).await {
            Ok(report) => {
                println!("{}", serde_json::to_string(&report).unwrap());
                if report["verdict"] == json!("gap") {
                    println!("PQNLD_CAPABILITY_GAP");
                }
            }
            Err(e) => {
                eprintln!("capability probe failed: {e}");
                println!("PQNLD_CAPABILITY_GAP");
            }
        }
        return;
    }

    let fixture = PathBuf::from(match flag_value(&args, "--fixture") {
        Some(path) => path,
        None => {
            eprintln!("--fixture is required (or use --self-test/--capability-probe/--summarize)");
            std::process::exit(1);
        }
    });
    let arm = flag_value(&args, "--arm").unwrap_or_else(|| "arm".to_string());
    let repeats: usize = flag_value(&args, "--repeats").and_then(|r| r.parse().ok()).unwrap_or(3);
    let assert = flag_present(&args, "--assert");

    let text = match std::fs::read_to_string(&fixture) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("cannot read {}: {e}", fixture.display());
            std::process::exit(1);
        }
    };

    let mut metrics = Vec::new();
    let mut refusals = 0usize;
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: Value = match serde_json::from_str(line) {
            Ok(row) => row,
            Err(e) => {
                eprintln!("fixture line {}: {e}", line_number + 1);
                std::process::exit(1);
            }
        };
        let state = row.get("state").cloned().unwrap_or(Value::Null);
        let Some(questions) = row.get("questions").and_then(Value::as_object) else { continue };
        let expected = row.get("expected");
        for (qkey, question) in questions {
            if question.get("type").and_then(Value::as_str) != Some("choice") {
                continue;
            }
            let run_id = row
                .get("_evaluation")
                .and_then(|e| e.get("run_id"))
                .and_then(Value::as_str)
                .unwrap_or("row");
            let result = run_question(
                &client,
                &state,
                &format!("{run_id}:{qkey}"),
                question,
                expected.and_then(|e| e.get(qkey)).and_then(Value::as_str).map(|s| s.to_string()),
                repeats,
            )
            .await;
            refusals += result.refusals.len();
            println!(
                "{} {qkey}: options={} label_samples={} position_samples={} refusals={}{}",
                now(),
                result.k,
                result.label_samples.len(),
                result.position_samples.len(),
                result.refusals.len(),
                if result.all_refusals() { " ALL-REFUSED" } else { "" }
            );
            metrics.push(question_metrics(&result, 0.0));
        }
    }

    let report = build_report(&arm, &fixture, metrics, refusals);
    let path = out.join(format!("arm-{arm}.json"));
    if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()) {
        eprintln!("cannot write {}: {e}", path.display());
        std::process::exit(1);
    }
    println!(
        "arm={arm} questions={} refusals={} spread_p95={} flip_rate={} inversions={} within-run verdict={}",
        report["questions"], report["refusals"], report["spread_p95"], report["flip_rate"],
        report["inversions"], report["verdict_within_run"]
    );

    if assert {
        let mut failures: Vec<String> = Vec::new();
        if report["questions"].as_u64().unwrap_or(0) == 0 {
            failures.push("no questions scored".to_string());
        }
        if report["refusals"].as_u64().unwrap_or(0) > 0 {
            failures.push(format!("{} refused samples", report["refusals"]));
        }
        for metric in report["metrics"].as_array().cloned().unwrap_or_default() {
            let floor = metric["repeat_floor"].as_f64().unwrap_or(0.0);
            if floor != 0.0 {
                failures.push(format!("{}: repeats differ by {floor}", metric["id"]));
            }
        }
        if failures.is_empty() {
            println!("PQNLD_ARM_ASSERTIONS_OK");
        } else {
            for failure in &failures {
                eprintln!("assertion failed: {failure}");
            }
            std::process::exit(1);
        }
    }
}
