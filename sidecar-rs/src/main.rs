// pqnld-rs — turn any served vLLM/OpenAI model into a typed decision engine.
//
// Drop-in replacement for the Python sidecar: renders a Decision Index question
// with letter-labelled options, reads the model's answer-slot distribution over
// those labels in one (or ceil(n/128)) chat requests, and serves the typed
// distribution at POST /v1/decide (alias /v1/systemone). A /v1/chat/completions
// shim answers decision JSON and returns a notice for ordinary chat.
//
// The readout route comes from a models/<name>.json descriptor (lettered|echo|
// auto). "auto" probes the model at startup and falls back to echo when the
// lettered readout does not hold. Option labels are tokenizer-verified single
// tokens; questions with more than 26 options relabel with an extended
// single-token alphabet and the shared prompt is split across requests when the
// engine's 128 explicit-token cap is exceeded.
use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::Infallible;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::{Mutex, Semaphore};

mod engine;

use engine::{Engine, EngineError};

const LETTERS: &str = "abcdefghijklmnopqrstuvwxyz";
const MAX_LOGPROB_IDS: usize = 128;
const MAX_QUESTION_WORKERS: usize = 1;
const DEFAULT_BATCH: usize = 16;
const DEFAULT_CACHE_SIZE: usize = 256;

const SYSTEM_PROMPT: &str = "You are a decision engine. Read the state, then answer the question by replying with exactly one option key from the list. Do not explain.";
const LETTER_SYSTEM_PROMPT: &str = "You are a decision engine. Read the state, then answer the question by replying with exactly one option letter from the list. Do not explain.";
const CHAT_TEMPLATE: &str = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n";
const PROBE_STATE: &str = "The sky is blue.";
const CHAT_NOTICE: &str = "This model is a typed decision engine. Send a JSON object with 'state' and 'questions' (Decision Index format) as the user message.";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Lettered,
    Echo,
}

fn default_readout() -> String {
    "auto".to_string()
}
fn default_letters() -> String {
    LETTERS.to_string()
}
fn default_temperature() -> f64 {
    1.0
}
fn default_system_prompt() -> String {
    SYSTEM_PROMPT.to_string()
}
fn default_letter_system_prompt() -> String {
    LETTER_SYSTEM_PROMPT.to_string()
}
fn default_chat_template() -> Option<String> {
    Some(CHAT_TEMPLATE.to_string())
}
fn default_engine_profile() -> String {
    "full".to_string()
}

#[derive(Clone, Serialize, Deserialize)]
struct ModelSpec {
    #[serde(default = "default_readout")]
    readout: String,
    #[serde(default = "default_letters")]
    letters: String,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default)]
    enable_thinking: bool,
    #[serde(default = "default_system_prompt")]
    system_prompt: String,
    #[serde(default = "default_letter_system_prompt")]
    letter_system_prompt: String,
    #[serde(default = "default_chat_template")]
    chat_template: Option<String>,
    #[serde(default = "default_engine_profile")]
    engine_profile: String,
    #[serde(default)]
    specific_token_scores: bool,
}

impl Default for ModelSpec {
    fn default() -> Self {
        ModelSpec {
            readout: default_readout(),
            letters: default_letters(),
            temperature: default_temperature(),
            enable_thinking: false,
            system_prompt: default_system_prompt(),
            letter_system_prompt: default_letter_system_prompt(),
            chat_template: default_chat_template(),
            engine_profile: default_engine_profile(),
            specific_token_scores: false,
        }
    }
}

struct Lru {
    map: HashMap<String, Value>,
    order: VecDeque<String>,
    cap: usize,
}

impl Lru {
    fn new(cap: usize) -> Self {
        Lru {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap,
        }
    }

    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.to_string());
    }

    fn get(&mut self, key: &str) -> Option<Value> {
        if self.map.contains_key(key) {
            self.touch(key);
            self.map.get(key).cloned()
        } else {
            None
        }
    }

    fn put(&mut self, key: String, value: Value) {
        if self.map.contains_key(&key) {
            self.touch(&key);
        } else {
            self.order.push_back(key.clone());
        }
        self.map.insert(key, value);
        while self.map.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

#[derive(Clone)]
struct Ctx {
    engine: Arc<dyn Engine>,
    model: String,
    spec: Arc<ModelSpec>,
    temperature: f64,
    batch: usize,
    extended: Arc<Vec<char>>,
    mode: Arc<Mutex<Option<Mode>>>,
    cache: Arc<StdMutex<Lru>>,
    label_ids: Arc<Mutex<HashMap<String, u32>>>,
    sem: Arc<Semaphore>,
    workers: usize,
}

fn extended_labels() -> Vec<char> {
    let mut v = Vec::new();
    for c in 'a'..='z' {
        v.push(c);
    }
    for c in 'A'..='Z' {
        v.push(c);
    }
    for c in '0'..='9' {
        v.push(c);
    }
    for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".chars() {
        v.push(c);
    }
    for cp in 0x391u32..0x3AA {
        if let Some(c) = char::from_u32(cp) {
            v.push(c);
        }
    }
    for cp in 0x3B1u32..0x3CA {
        if let Some(c) = char::from_u32(cp) {
            v.push(c);
        }
    }
    for cp in 0x410u32..0x450 {
        if let Some(c) = char::from_u32(cp) {
            v.push(c);
        }
    }
    v
}

fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => {
            let mut buf = Vec::new();
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, PyFormatter);
            if other.serialize(&mut ser).is_err() {
                return serde_json::to_string(other).unwrap_or_default();
            }
            String::from_utf8(buf).unwrap_or_default()
        }
    }
}

/// Render an f64 the way Python's `repr`/`json.dumps` does, so a state round
/// tripped through Python and Rust produces byte-identical prompts.
fn py_float(f: f64) -> String {
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let sci = format!("{:e}", f);
    let (mant, exp) = match sci.split_once('e') {
        Some(parts) => parts,
        None => return sci,
    };
    let exp: i32 = exp.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| *c != '-' && *c != '.').collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let e = exp as usize;
            if digits.len() > e + 1 {
                out.push_str(&digits[..e + 1]);
                out.push('.');
                out.push_str(&digits[e + 1..]);
            } else {
                out.push_str(&digits);
                for _ in digits.len()..=e {
                    out.push('0');
                }
                out.push_str(".0");
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
    out
}

struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(py_float(value).as_bytes())
    }
}

fn is_empty_state(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

fn letter_of(token: &str) -> Option<String> {
    let t = token.trim().to_lowercase();
    if t.chars().count() == 1 && LETTERS.contains(&t) {
        Some(t)
    } else {
        None
    }
}

async fn label_id(ctx: &Ctx, label: &str, required: bool) -> Result<Option<u32>, EngineError> {
    if let Some(id) = ctx.label_ids.lock().await.get(label) {
        return Ok(Some(*id));
    }
    if !ctx.engine.capabilities().tokenize {
        return if required {
            Err(EngineError::Unsupported(
                "engine has no tokenize endpoint".to_string(),
            ))
        } else {
            Ok(None)
        };
    }
    let tokens = ctx.engine.tokenize(label).await?;
    let id = if tokens.len() == 1 { Some(tokens[0]) } else { None };
    match id {
        Some(i) => {
            ctx.label_ids.lock().await.insert(label.to_string(), i);
            Ok(Some(i))
        }
        None if required => Err(EngineError::Unsupported(format!(
            "option label {label:?} is not a single token"
        ))),
        None => Ok(None),
    }
}

async fn labels(ctx: &Ctx, count: usize) -> Result<Vec<String>, EngineError> {
    let letters: Vec<char> = ctx.spec.letters.chars().collect();
    if count <= letters.len() {
        return Ok(letters[..count].iter().map(|c| c.to_string()).collect());
    }
    let mut out = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    for c in ctx.extended.iter() {
        if out.len() == count {
            break;
        }
        let s = c.to_string();
        if let Some(id) = label_id(ctx, &s, false).await? {
            if seen.insert(id) {
                out.push(s);
            }
        }
    }
    if out.len() < count {
        return Err(EngineError::Unsupported(format!(
            "tokenizer has only {} single-token option labels, need {count}",
            out.len()
        )));
    }
    Ok(out)
}

fn render_lettered(state: &Value, question: &Value, keys: &[String], labels: &[String]) -> String {
    let mut lines = vec!["State:".to_string()];
    lines.push(if is_empty_state(state) {
        "(empty)".to_string()
    } else {
        text(state)
    });
    lines.push(String::new());
    lines.push("Question:".to_string());
    let instructions = question
        .get("instructions")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    lines.push(text(&instructions));
    lines.push(String::new());
    lines.push("Options:".to_string());
    let criteria = question.get("criteria").and_then(|c| c.as_object());
    for (label, key) in labels.iter().zip(keys.iter()) {
        let description = criteria.and_then(|m| m.get(key));
        let rendered = match description {
            Some(Value::Null) | None => key.clone(),
            Some(v) => text(v),
        };
        lines.push(format!("{label}) {rendered}"));
    }
    lines.push(String::new());
    lines.push("Reply with exactly one option letter.".to_string());
    lines.join("\n")
}

fn render_prompt(state: &Value, question: &Value) -> String {
    let mut lines = vec!["State:".to_string()];
    lines.push(if is_empty_state(state) {
        "(empty)".to_string()
    } else {
        text(state)
    });
    lines.push(String::new());
    lines.push("Question:".to_string());
    let instructions = question
        .get("instructions")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    lines.push(text(&instructions));
    lines.push(String::new());
    let qtype = question.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let criteria: Vec<(String, Value)> = if qtype == "choice" {
        question
            .get("criteria")
            .and_then(|c| c.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default()
    } else {
        vec![
            ("false".to_string(), Value::String("False".to_string())),
            ("true".to_string(), Value::String("True".to_string())),
        ]
    };
    lines.push("Options:".to_string());
    for (key, description) in criteria {
        let rendered = if description.is_null() {
            key.clone()
        } else {
            text(&description)
        };
        lines.push(format!("{key}: {rendered}"));
    }
    lines.push(String::new());
    lines.push("Reply with exactly one option key.".to_string());
    lines.join("\n")
}

async fn letter_scores(
    ctx: &Ctx,
    state: &Value,
    question: &Value,
    keys: &[String],
) -> Result<(Vec<f64>, i64, Option<String>), EngineError> {
    let option_labels = labels(ctx, keys.len()).await?;
    let body = render_lettered(state, question, keys, &option_labels);
    let capabilities = ctx.engine.capabilities();
    let specific = ctx.spec.specific_token_scores && capabilities.exact_ids;

    let mut id_by_label: HashMap<String, u32> = HashMap::new();
    let mut label_by_id: HashMap<u32, String> = HashMap::new();
    if specific {
        for label in &option_labels {
            let id = label_id(ctx, label, true).await?.ok_or_else(|| {
                EngineError::Unsupported(format!("option label {label:?} is not a single token"))
            })?;
            id_by_label.insert(label.clone(), id);
            label_by_id.insert(id, label.clone());
        }
        let distinct: HashSet<u32> = id_by_label.values().copied().collect();
        if distinct.len() != option_labels.len() {
            return Err(EngineError::Unsupported(
                "option labels do not have distinct token IDs".to_string(),
            ));
        }
    }

    let chunks: Vec<Vec<String>> = if specific {
        option_labels
            .chunks(MAX_LOGPROB_IDS)
            .map(|c| c.to_vec())
            .collect()
    } else {
        vec![option_labels.clone()]
    };

    let mut by_label: HashMap<String, f64> = HashMap::new();
    let mut usage_tokens: i64 = 0;
    let mut top_token: Option<String> = None;
    for chunk in &chunks {
        if specific {
            let ids: Vec<u32> = chunk.iter().map(|label| id_by_label[label]).collect();
            let scores = ctx.engine.score_token_ids(&body, &ids).await?;
            if top_token.is_none() {
                if let Some((id, _)) = scores.first() {
                    top_token = label_by_id.get(id).cloned();
                }
            }
            for (id, logprob) in scores {
                if let Some(label) = label_by_id.get(&id) {
                    by_label.entry(label.clone()).or_insert(logprob);
                }
            }
        } else {
            let scores = ctx
                .engine
                .topk(&body, std::cmp::max(keys.len(), 20))
                .await?;
            if top_token.is_none() {
                let mut best: Option<(f64, String)> = None;
                for (token, logprob) in &scores {
                    if best.as_ref().map(|(b, _)| *logprob > *b).unwrap_or(true) {
                        best = Some((*logprob, token.clone()));
                    }
                }
                top_token = best.map(|(_, token)| token);
            }
            for (token, logprob) in scores {
                if let Some(label) = letter_of(&token) {
                    if option_labels.contains(&label) {
                        by_label.entry(label).or_insert(logprob);
                    }
                }
            }
        }
        usage_tokens = ctx.engine.last_usage();
    }

    let missing: Vec<String> = option_labels
        .iter()
        .filter(|l| !by_label.contains_key(*l))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(EngineError::Unsupported(format!(
            "answer-slot logprobs missing option labels: {}",
            missing.join(", ")
        )));
    }
    let scores: Vec<f64> = option_labels.iter().map(|l| by_label[l]).collect();
    Ok((scores, usage_tokens, top_token))
}

async fn echo_scores(ctx: &Ctx, context: &str, keys: &[String]) -> Result<(Vec<f64>, i64), EngineError> {
    let mut ordered: Vec<String> = keys.to_vec();
    ordered.sort();
    for pair in ordered.windows(2) {
        if pair[1].starts_with(&pair[0]) {
            return Err(EngineError::Unsupported(
                "echo scoring cannot compare prefix-overlapping option keys".to_string(),
            ));
        }
    }
    if !ctx.engine.capabilities().prompt_logprobs {
        return Err(EngineError::Unsupported(
            "this engine does not expose prompt/continuation logprobs".to_string(),
        ));
    }
    let context_pairs = ctx.engine.prompt_logprobs(context).await?;
    let ctx_tokens: Vec<String> = context_pairs.into_iter().map(|(token, _)| token).collect();
    let mut scores = Vec::new();
    let mut usage_tokens = 0i64;
    for chunk in keys.chunks(ctx.batch) {
        let prompts: Vec<String> = chunk.iter().map(|k| format!("{context}{k}")).collect();
        let batch = ctx.engine.prompt_logprobs_batch(&prompts).await?;
        usage_tokens = ctx.engine.last_usage();
        for pairs in batch {
            let tokens: Vec<String> = pairs.iter().map(|(token, _)| token.clone()).collect();
            let mut shared = 0;
            while shared < ctx_tokens.len()
                && shared < tokens.len()
                && tokens[shared] == ctx_tokens[shared]
            {
                shared += 1;
            }
            let start = shared.min(pairs.len());
            scores.push(pairs[start..].iter().map(|(_, logprob)| logprob).sum());
        }
    }
    Ok((scores, usage_tokens))
}

fn echo_context(ctx: &Ctx, state: &Value, question: &Value) -> Result<String, EngineError> {
    let template = ctx.spec.chat_template.as_ref().ok_or_else(|| {
        EngineError::Unsupported(
            "this question needs the echo readout but the model has no chat_template".to_string(),
        )
    })?;
    Ok(template
        .replace("{system}", &ctx.spec.system_prompt)
        .replace("{body}", &render_prompt(state, question)))
}

fn require_template(ctx: &Ctx) -> Result<(), EngineError> {
    if ctx.spec.chat_template.is_none() {
        Err(EngineError::Unsupported(format!(
            "the echo readout for {:?} requires a chat_template",
            ctx.model
        )))
    } else {
        Ok(())
    }
}

async fn probe(ctx: &Ctx) -> Result<Mode, EngineError> {
    let probe_question = json!({
        "type": "choice",
        "instructions": "Which colour is named?",
        "criteria": {"red": "red", "blue": "blue", "green": "green"},
    });
    let keys: Vec<String> = vec!["red".to_string(), "blue".to_string(), "green".to_string()];
    let lettered = match letter_scores(ctx, &Value::String(PROBE_STATE.to_string()), &probe_question, &keys).await {
        Ok((_, _, Some(top))) => letter_of(&top)
            .map(|l| ctx.spec.letters.contains(&l))
            .unwrap_or(false),
        Ok((_, _, None)) => false,
        Err(e @ EngineError::Unsupported(_)) => {
            if ctx.spec.specific_token_scores {
                return Err(e);
            }
            false
        }
        Err(EngineError::Other(_)) => false,
    };
    if lettered {
        Ok(Mode::Lettered)
    } else {
        require_template(ctx)?;
        Ok(Mode::Echo)
    }
}

async fn ensure_mode(ctx: &Ctx) -> Result<Mode, EngineError> {
    if let Some(mode) = *ctx.mode.lock().await {
        return Ok(mode);
    }
    let mode = match ctx.spec.readout.as_str() {
        "lettered" => Mode::Lettered,
        "echo" => {
            require_template(ctx)?;
            Mode::Echo
        }
        "auto" => probe(ctx).await?,
        other => return Err(EngineError::Unsupported(format!("unknown readout {other:?}"))),
    };
    *ctx.mode.lock().await = Some(mode);
    Ok(mode)
}

async fn score_question(
    ctx: Arc<Ctx>,
    state: Value,
    key: String,
    question: Value,
) -> Result<(String, Value, i64), EngineError> {
    let mode = ensure_mode(&ctx).await?;
    let qtype = question
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_string();
    let keys: Vec<String> = if qtype == "choice" {
        question
            .get("criteria")
            .and_then(|c| c.as_object())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    } else if qtype == "noul" {
        vec!["false".to_string(), "true".to_string()]
    } else {
        return Err(EngineError::Unsupported(format!(
            "{key}: unsupported question type {qtype:?}"
        )));
    };
    let letter_count = ctx.spec.letters.chars().count();
    let (scores, usage_tokens, _top) = if mode == Mode::Lettered
        && (keys.len() <= letter_count || ctx.spec.specific_token_scores)
    {
        letter_scores(&ctx, &state, &question, &keys).await?
    } else {
        let context = echo_context(&ctx, &state, &question)?;
        let (s, u) = echo_scores(&ctx, &context, &keys).await?;
        (s, u, None)
    };
    let peak = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores
        .iter()
        .map(|s| ((s - peak) / ctx.temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|w| w / total).collect();

    let answer = if qtype == "choice" {
        let mut best = 0usize;
        for (i, p) in probs.iter().enumerate() {
            if *p > probs[best] {
                best = i;
            }
        }
        let mut probabilities = Map::new();
        for (k, p) in keys.iter().zip(probs.iter()) {
            probabilities.insert(k.clone(), json!(p));
        }
        json!({
            "type": "choice",
            "choice": keys[best],
            "probabilities": Value::Object(probabilities),
        })
    } else {
        let idx = keys.iter().position(|k| k == "true").unwrap_or(1);
        json!({"type": "noul", "noul": probs[idx]})
    };
    Ok((key, answer, usage_tokens))
}

fn validate(questions: &Map<String, Value>, response: &Value) -> Result<(), String> {
    let answers = response
        .get("answers")
        .and_then(|a| a.as_object())
        .ok_or("no answers object")?;
    if answers.len() != questions.len() || !questions.keys().all(|k| answers.contains_key(k)) {
        return Err("answer keys do not match question keys".to_string());
    }
    for (key, question) in questions {
        let answer = &answers[key];
        let qtype = question.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if answer.get("type").and_then(|t| t.as_str()) != Some(qtype) {
            return Err(format!("{key}: type mismatch"));
        }
        if qtype == "choice" {
            let criteria = question
                .get("criteria")
                .and_then(|c| c.as_object())
                .ok_or("choice question has no criteria")?;
            let choice = answer.get("choice").and_then(|c| c.as_str()).unwrap_or("");
            if !criteria.contains_key(choice) {
                return Err(format!("{key}: choice outside criteria"));
            }
            let probs = answer
                .get("probabilities")
                .and_then(|p| p.as_object())
                .ok_or("choice answer has no probabilities")?;
            if probs.len() != criteria.len() || !criteria.keys().all(|k| probs.contains_key(k)) {
                return Err(format!("{key}: probability keys do not match criteria"));
            }
            let mut sum = 0.0f64;
            for value in probs.values() {
                match value.as_f64() {
                    Some(p) if p.is_finite() && (0.0..=1.0).contains(&p) => sum += p,
                    _ => return Err(format!("{key}: probability out of range")),
                }
            }
            if (sum - 1.0).abs() > 0.01 {
                return Err(format!("{key}: probabilities do not sum to 1"));
            }
        } else if qtype == "noul" {
            match answer.get("noul").and_then(|v| v.as_f64()) {
                Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) => {}
                _ => return Err(format!("{key}: noul out of range")),
            }
        } else {
            return Err(format!("{key}: unsupported question type"));
        }
    }
    Ok(())
}

fn cache_key(ctx: &Ctx, mode: Mode, state: &Value, questions: &Map<String, Value>) -> String {
    let spec = serde_json::to_value(&*ctx.spec).unwrap_or(Value::Null);
    let readout = match mode {
        Mode::Lettered => "lettered",
        Mode::Echo => "echo",
    };
    serde_json::to_string(&json!({
        "model": ctx.model,
        "readout": readout,
        "spec": spec,
        "temperature": ctx.temperature,
        "state": state,
        "questions": questions,
    }))
    .unwrap_or_default()
}

async fn readout_call(
    ctx: Arc<Ctx>,
    state: &Value,
    questions: &Map<String, Value>,
) -> Result<Value, EngineError> {
    let mode = ensure_mode(&ctx).await?;
    let key = cache_key(&ctx, mode, state, questions);
    if let Some(hit) = ctx.cache.lock().unwrap().get(&key) {
        return Ok(hit);
    }
    let items: Vec<(String, Value)> = questions
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut answers = Map::new();
    let mut input_tokens = 0i64;
    if ctx.workers == 1 {
        for (question_key, question) in items {
            let (question_key, answer, tokens) =
                score_question(ctx.clone(), state.clone(), question_key, question).await?;
            answers.insert(question_key, answer);
            input_tokens += tokens;
        }
    } else {
        let mut handles = Vec::new();
        for (question_key, question) in items {
            let ctx = ctx.clone();
            let state = state.clone();
            handles.push(tokio::spawn(async move {
                let _permit = ctx.sem.acquire().await;
                score_question(ctx.clone(), state, question_key, question).await
            }));
        }
        for handle in handles {
            match handle.await {
                Ok(Ok((question_key, answer, tokens))) => {
                    answers.insert(question_key, answer);
                    input_tokens += tokens;
                }
                Ok(Err(error)) => return Err(error),
                Err(e) => return Err(EngineError::Other(format!("readout task failed: {e}"))),
            }
        }
    }
    let response = json!({
        "model": ctx.model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens},
    });
    if let Err(e) = validate(questions, &response) {
        return Err(EngineError::Other(format!("readout self-check failed: {e}")));
    }
    ctx.cache.lock().unwrap().put(key, response.clone());
    Ok(response)
}

fn decision_from_messages(messages: &Value) -> Option<(Value, Map<String, Value>)> {
    let arr = messages.as_array()?;
    let content = arr
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))?
        .get("content")?
        .as_str()?;
    let payload: Value = serde_json::from_str(content).ok()?;
    let questions = payload.get("questions")?.as_object()?.clone();
    let state = payload
        .get("state")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    Some((state, questions))
}

fn json_response(code: u16, value: Value) -> Response<Full<Bytes>> {
    let body = serde_json::to_vec(&value).unwrap_or_default();
    Response::builder()
        .status(StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .expect("valid response")
}

fn error_response(code: u16, message: String) -> Response<Full<Bytes>> {
    json_response(code, json!({ "error": message }))
}

fn engine_error_response(error: EngineError) -> Response<Full<Bytes>> {
    match error {
        EngineError::Unsupported(message) => error_response(422, message),
        EngineError::Other(message) => error_response(500, message),
    }
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

async fn handle_decide(ctx: Arc<Ctx>, body: Bytes) -> Response<Full<Bytes>> {
    let payload: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(e) => return error_response(400, format!("bad request: {e}")),
    };
    let state = payload
        .get("state")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    let questions = match payload.get("questions").and_then(|q| q.as_object()) {
        Some(map) => map.clone(),
        None => return error_response(400, "bad request: missing questions".to_string()),
    };
    let mut response = match readout_call(ctx.clone(), &state, &questions).await {
        Ok(response) => response,
        Err(error) => return engine_error_response(error),
    };
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&ctx.model);
    response["model"] = Value::String(model.to_string());
    json_response(200, response)
}

async fn handle_chat(ctx: Arc<Ctx>, body: Bytes) -> Response<Full<Bytes>> {
    let payload: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(e) => return error_response(400, format!("bad request: {e}")),
    };
    let messages = match payload.get("messages") {
        Some(messages) => messages.clone(),
        None => return error_response(400, "bad request: missing messages".to_string()),
    };
    let (content, tokens) = match decision_from_messages(&messages) {
        None => (CHAT_NOTICE.to_string(), 0i64),
        Some((state, questions)) => match readout_call(ctx.clone(), &state, &questions).await {
            Ok(response) => {
                let toks = response
                    .get("usage")
                    .and_then(|u| u.get("input_tokens"))
                    .and_then(|t| t.as_i64())
                    .unwrap_or(0);
                (serde_json::to_string(&response).unwrap_or_default(), toks)
            }
            Err(error) => return engine_error_response(error),
        },
    };
    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or(&ctx.model)
        .to_string();
    let completion = json!({
        "id": "chatcmpl-decision",
        "object": "chat.completion",
        "created": now_epoch(),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
    });
    if payload.get("stream").and_then(|s| s.as_bool()).unwrap_or(false) {
        return stream_response(&completion);
    }
    json_response(200, completion)
}

fn stream_response(completion: &Value) -> Response<Full<Bytes>> {
    let choice = &completion["choices"][0];
    let chunks = vec![
        json!({
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": choice["message"]["content"]}, "finish_reason": null}],
        }),
        json!({
            "id": completion["id"],
            "object": "chat.completion.chunk",
            "created": completion["created"],
            "model": completion["model"],
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": completion["usage"],
        }),
    ];
    let mut body = String::new();
    for chunk in chunks {
        body.push_str(&format!(
            "data: {}\n\n",
            serde_json::to_string(&chunk).unwrap_or_default()
        ));
    }
    body.push_str("data: [DONE]\n\n");
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .body(Full::new(Bytes::from(body)))
        .expect("valid response")
}

async fn route(req: Request<Incoming>, ctx: Arc<Ctx>) -> Response<Full<Bytes>> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    match (method.as_str(), path.as_str()) {
        ("GET", "/healthz") => json_response(200, json!({"status": "ok"})),
        ("POST", "/v1/systemone") | ("POST", "/v1/decide") => match req.into_body().collect().await {
            Ok(body) => handle_decide(ctx, body.to_bytes()).await,
            Err(e) => error_response(400, format!("bad request: {e}")),
        },
        ("POST", "/v1/chat/completions") => match req.into_body().collect().await {
            Ok(body) => handle_chat(ctx, body.to_bytes()).await,
            Err(e) => error_response(400, format!("bad request: {e}")),
        },
        _ => error_response(404, "not found".to_string()),
    }
}

async fn serve_connection<I>(io: I, ctx: Arc<Ctx>)
where
    I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
{
    let service = service_fn(move |req| {
        let ctx = ctx.clone();
        async move { Ok::<_, Infallible>(route(req, ctx).await) }
    });
    let _ = hyper::server::conn::http1::Builder::new()
        .serve_connection(TokioIo::new(io), service)
        .await;
}

async fn accept_tcp(listener: TcpListener, ctx: Arc<Ctx>) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let ctx = ctx.clone();
                tokio::spawn(async move { serve_connection(stream, ctx).await });
            }
            Err(e) => eprintln!("tcp accept error: {e}"),
        }
    }
}

async fn accept_uds(listener: UnixListener, ctx: Arc<Ctx>) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                let ctx = ctx.clone();
                tokio::spawn(async move { serve_connection(stream, ctx).await });
            }
            Err(e) => eprintln!("uds accept error: {e}"),
        }
    }
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn env_first(primary: &str, legacy: &str) -> Option<String> {
    std::env::var(primary)
        .ok()
        .or_else(|| std::env::var(legacy).ok())
}

fn load_spec(models_dir: &str, name: &str) -> ModelSpec {
    let path = format!("{models_dir}/{name}.json");
    match std::fs::read_to_string(&path) {
        Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
        Err(_) => ModelSpec::default(),
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let engine_url = arg_value(&args, "--vllm-url")
        .or_else(|| env_first("PQNLD_VLLM_URL", "DECISION_VLLM_URL"))
        .unwrap_or_else(|| "http://127.0.0.1:11542".to_string());
    let engine_url = engine_url.trim_end_matches('/').to_string();
    let model = arg_value(&args, "--model")
        .or_else(|| env_first("PQNLD_MODEL", "DECISION_MODEL"))
        .unwrap_or_else(|| "qwen38-27b".to_string());
    let descriptor = arg_value(&args, "--descriptor")
        .or_else(|| env_first("PQNLD_DESCRIPTOR", "DECISION_DESCRIPTOR"));
    let models_dir = arg_value(&args, "--models-dir")
        .or_else(|| env_first("PQNLD_MODELS_DIR", "DECISION_MODELS_DIR"))
        .unwrap_or_else(|| "models".to_string());
    let host = arg_value(&args, "--host")
        .or_else(|| env_first("PQNLD_HOST", "DECISION_HOST"))
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = arg_value(&args, "--port")
        .or_else(|| env_first("PQNLD_PORT", "DECISION_PORT"))
        .and_then(|p| p.parse().ok())
        .unwrap_or(11560);
    let uds = arg_value(&args, "--uds");
    let timeout: u64 = arg_value(&args, "--timeout")
        .and_then(|t| t.parse().ok())
        .unwrap_or(600);
    let workers: usize = arg_value(&args, "--workers")
        .and_then(|w| w.parse().ok())
        .unwrap_or(MAX_QUESTION_WORKERS)
        .max(1);

    let spec = load_spec(&models_dir, descriptor.as_deref().unwrap_or(&model));
    let temperature = arg_value(&args, "--temperature")
        .and_then(|t| t.parse().ok())
        .or_else(|| {
            env_first("PQNLD_TEMPERATURE", "DECISION_TEMPERATURE").and_then(|t| t.parse().ok())
        })
        .or(Some(spec.temperature))
        .unwrap_or(1.0);
    if temperature <= 0.0 {
        panic!("temperature must be > 0");
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout))
        .build()
        .expect("http client");
    let engine_kind = arg_value(&args, "--engine-kind")
        .or_else(|| env_first("PQNLD_ENGINE_KIND", "DECISION_ENGINE_KIND"))
        .unwrap_or_else(|| "vllm".to_string());
    let kind = if engine_kind == "auto" {
        engine::detect_kind(&client, &engine_url).await
    } else {
        match engine::parse_kind(&engine_kind) {
            Some(kind) => kind,
            None => {
                eprintln!("unknown engine kind {engine_kind:?} (vllm|llamacpp|sglang|openai|auto)");
                std::process::exit(1);
            }
        }
    };
    let engine = engine::build(
        kind,
        client,
        engine_url.clone(),
        model.clone(),
        spec.letter_system_prompt.clone(),
        spec.enable_thinking,
    );
    let ctx = Arc::new(Ctx {
        engine,
        model: model.clone(),
        spec: Arc::new(spec),
        temperature,
        batch: DEFAULT_BATCH,
        extended: Arc::new(extended_labels()),
        mode: Arc::new(Mutex::new(None)),
        cache: Arc::new(StdMutex::new(Lru::new(DEFAULT_CACHE_SIZE))),
        label_ids: Arc::new(Mutex::new(HashMap::new())),
        sem: Arc::new(Semaphore::new(workers)),
        workers,
    });

    let mode = match ensure_mode(&ctx).await {
        Ok(mode) => mode,
        Err(EngineError::Unsupported(message)) | Err(EngineError::Other(message)) => {
            eprintln!("startup failed: {message}");
            std::process::exit(1);
        }
    };
    let mode_name = match mode {
        Mode::Lettered => "lettered",
        Mode::Echo => "echo",
    };

    if let Some(uds_path) = uds.clone() {
        let _ = std::fs::remove_file(&uds_path);
        let listener = UnixListener::bind(&uds_path).expect("bind unix socket");
        eprintln!("pqnld-rs listening on unix:{uds_path} -> {engine_url} (readout={mode_name})");
        let ctx = ctx.clone();
        tokio::spawn(async move { accept_uds(listener, ctx).await });
    }

    let bind = format!("{host}:{port}");
    let listener = TcpListener::bind(&bind).await.expect("bind tcp");
    eprintln!(
        "decision readout on http://{bind}/v1/decide -> {engine_url} (readout={mode_name})"
    );
    let ctx = ctx.clone();
    tokio::spawn(async move { accept_tcp(listener, ctx).await });

    std::future::pending::<()>().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_float_matches_python_repr() {
        for (value, expected) in [
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (7.249627201966407e-05, "7.249627201966407e-05"),
            (0.00020145927898402295, "0.00020145927898402295"),
            (2.0, "2.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (123.456, "123.456"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(py_float(value), expected, "value {value}");
        }
    }

    #[test]
    fn text_serializes_json_like_python() {
        assert_eq!(text(&json!({"a": 0.0001})), r#"{"a":0.0001}"#);
        assert_eq!(text(&json!({"a": 7.249627201966407e-05})), r#"{"a":7.249627201966407e-05}"#);
        assert_eq!(text(&json!("plain")), "plain");
    }
}
