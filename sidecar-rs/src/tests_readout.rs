//! Rust ports of the readout invariants previously covered by the Python unit
//! tests. A `MockEngine` and holding `Ctx` let these run without a live engine.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{json, Map, Value};
use tokio::sync::{Mutex, Semaphore};

use super::engine::{BoxFuture, Capabilities, Engine, EngineError, Surface};
use super::*;

struct MockEngine {
    caps: Capabilities,
    tokenize_map: HashMap<String, Vec<u32>>,
    scores: HashMap<u32, f64>,
    only_known: bool,
    topk: Vec<(String, f64)>,
    prompt: Vec<(String, f64)>,
    sampled: u32,
    calls: Mutex<Vec<Vec<u32>>>,
}

fn caps(exact_ids: bool, top_k: bool, prompt_logprobs: bool) -> Capabilities {
    Capabilities {
        tokenize: true,
        exact_ids,
        top_k,
        prompt_logprobs,
        max_top_k: Some(20),
        max_explicit_ids: Some(MAX_LOGPROB_IDS),
        surface: Surface::ChatCompletions,
    }
}

impl MockEngine {
    fn exact(scores: &[(u32, f64)]) -> Self {
        MockEngine {
            caps: caps(true, false, false),
            tokenize_map: HashMap::new(),
            scores: scores.iter().copied().collect(),
            only_known: false,
            topk: Vec::new(),
            prompt: Vec::new(),
            sampled: 0,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn exact_strict(scores: &[(u32, f64)]) -> Self {
        MockEngine {
            only_known: true,
            ..MockEngine::exact(scores)
        }
    }

    fn topk_only(tokens: &[(&str, f64)]) -> Self {
        MockEngine {
            caps: caps(false, true, false),
            tokenize_map: HashMap::new(),
            scores: HashMap::new(),
            only_known: false,
            topk: tokens.iter().map(|(t, s)| (t.to_string(), *s)).collect(),
            prompt: Vec::new(),
            sampled: 0,
            calls: Mutex::new(Vec::new()),
        }
    }

    fn echo(tokens: &[(String, f64)]) -> Self {
        MockEngine {
            caps: caps(false, false, true),
            tokenize_map: HashMap::new(),
            scores: HashMap::new(),
            only_known: false,
            topk: Vec::new(),
            prompt: tokens.to_vec(),
            sampled: 0,
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl Engine for MockEngine {
    fn capabilities(&self) -> Capabilities {
        self.caps
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>> {
        let ids = if let Some(ids) = self.tokenize_map.get(text) {
            ids.clone()
        } else if text.chars().count() == 1 {
            vec![text.chars().next().unwrap() as u32]
        } else {
            vec![0, 0]
        };
        Box::pin(async move { Ok(ids) })
    }

    fn score_token_ids<'a>(
        &'a self,
        _prompt: &'a str,
        ids: &'a [u32],
    ) -> BoxFuture<'a, Result<Vec<(u32, f64)>, EngineError>> {
        let requested = ids.to_vec();
        let mut out = vec![(self.sampled, -20.0)];
        for id in ids {
            if let Some(score) = self.scores.get(id) {
                out.push((*id, *score));
            } else if !self.only_known {
                out.push((*id, -((*id % 13) as f64)));
            }
        }
        Box::pin(async move {
            self.calls.lock().await.push(requested);
            Ok(out)
        })
    }

    fn topk<'a>(&'a self, _prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        let mut out = self.topk.clone();
        out.truncate(k);
        Box::pin(async move { Ok(out) })
    }

    fn prompt_logprobs<'a>(&'a self, _prompt: &'a str) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        let pairs = self.prompt.clone();
        Box::pin(async move { Ok(pairs) })
    }
}

fn make_ctx(engine: MockEngine, spec: ModelSpec) -> (Arc<Ctx>, Arc<MockEngine>) {
    let engine = Arc::new(engine);
    let ctx = Arc::new(Ctx {
        engine: engine.clone(),
        model: "mock".to_string(),
        temperature: spec.temperature,
        spec: Arc::new(spec),
        batch: DEFAULT_BATCH,
        extended: Arc::new(extended_labels()),
        mode: Arc::new(Mutex::new(None)),
        cache: Arc::new(StdMutex::new(Lru::new(DEFAULT_CACHE_SIZE))),
        label_ids: Arc::new(Mutex::new(HashMap::new())),
        sem: Arc::new(Semaphore::new(1)),
        workers: 1,
    });
    (ctx, engine)
}

fn choice(instructions: &str, criteria: &[&str]) -> Value {
    let mut map = Map::new();
    for key in criteria {
        map.insert((*key).to_string(), Value::Null);
    }
    json!({"type": "choice", "instructions": instructions, "criteria": Value::Object(map)})
}

fn letters_spec(specific: bool) -> ModelSpec {
    ModelSpec {
        readout: "lettered".to_string(),
        specific_token_scores: specific,
        ..Default::default()
    }
}

#[test]
fn letter_of_accepts_only_single_letters() {
    assert_eq!(letter_of("A").as_deref(), Some("a"));
    assert_eq!(letter_of(" a ").as_deref(), Some("a"));
    assert_eq!(letter_of("apple"), None);
    assert_eq!(letter_of("ab"), None);
    assert_eq!(letter_of("1"), None);
}

#[test]
fn extended_labels_are_unique_and_large() {
    let labels = extended_labels();
    assert!(labels.len() >= 206, "got {}", labels.len());
    let unique: std::collections::HashSet<char> = labels.iter().copied().collect();
    assert_eq!(unique.len(), labels.len());
}

#[test]
fn render_lettered_shape_and_empty_state() {
    let body = render_lettered(
        &Value::String(String::new()),
        &choice("Pick", &["red", "blue"]),
        &["red".to_string(), "blue".to_string()],
        &["a".to_string(), "b".to_string()],
    );
    assert!(body.contains("State:\n(empty)"));
    assert!(body.contains("Question:\nPick"));
    assert!(body.contains("a) red"));
    assert!(body.contains("b) blue"));
    assert!(body.trim_end().ends_with("Reply with exactly one option letter."));
}

#[test]
fn validate_rejects_bad_distributions() {
    let questions = json!({"q": {"type": "choice", "criteria": {"a": null, "b": null}}})
        .as_object()
        .cloned()
        .unwrap();
    let good = json!({"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}}}});
    assert!(validate(&questions, &good).is_ok());
    let wrong_keys =
        json!({"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}}}});
    assert!(validate(&questions, &wrong_keys).is_err());
    let out_of_range = json!({"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.5, "b": -0.5}}}});
    assert!(validate(&questions, &out_of_range).is_err());
    let missing = json!({"answers": {}});
    assert!(validate(&questions, &missing).is_err());
}

#[tokio::test]
async fn specific_scores_produce_a_distribution() {
    let (ctx, _engine) = make_ctx(
        MockEngine::exact(&[(97, -0.5), (98, -1.0), (99, -2.0)]),
        letters_spec(true),
    );
    let (_, answer, _) = score_question(
        ctx,
        Value::String("x".to_string()),
        "q".to_string(),
        choice("Pick", &["a", "b", "c"]),
    )
    .await
    .unwrap();
    assert_eq!(answer["type"], "choice");
    assert_eq!(answer["choice"], "a");
    let probs = answer["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), 3);
    let sum: f64 = probs.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-9, "sum {sum}");
}

#[tokio::test]
async fn specific_scores_refuse_missing_labels() {
    let (ctx, _engine) = make_ctx(
        MockEngine::exact_strict(&[(97, -0.5), (98, -1.0)]),
        letters_spec(true),
    );
    let error = score_question(
        ctx,
        Value::String("x".to_string()),
        "q".to_string(),
        choice("Pick", &["a", "b", "c"]),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, EngineError::Unsupported(_)));
}

#[tokio::test]
async fn noul_reports_true_probability() {
    let (ctx, _engine) = make_ctx(
        MockEngine::exact(&[(97, -3.0), (98, -0.1)]),
        letters_spec(true),
    );
    let (_, answer, _) = score_question(
        ctx,
        Value::String("x".to_string()),
        "q".to_string(),
        json!({"type": "noul", "instructions": "Is it urgent?"}),
    )
    .await
    .unwrap();
    assert_eq!(answer["type"], "noul");
    let value = answer["noul"].as_f64().unwrap();
    assert!(value > 0.9 && value <= 1.0, "noul {value}");
}

#[tokio::test]
async fn over_128_options_split_and_merge_losslessly() {
    let count = 151usize;
    let keys: Vec<String> = (0..count).map(|i| format!("k{i:03}")).collect();
    let mut criteria = Map::new();
    for key in &keys {
        criteria.insert(key.clone(), Value::Null);
    }
    let question = json!({
        "type": "choice",
        "instructions": "Pick one",
        "criteria": Value::Object(criteria),
    });
    let (ctx, engine) = make_ctx(MockEngine::exact(&[]), letters_spec(true));
    let (_, answer, _) = score_question(ctx, Value::String("x".to_string()), "q".to_string(), question)
        .await
        .unwrap();
    let probs = answer["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), count);
    let calls = engine.calls.lock().await;
    assert_eq!(calls.len(), 2, "expected two id chunks");
    assert_eq!(calls[0].len(), MAX_LOGPROB_IDS);
    assert_eq!(calls[1].len(), count - MAX_LOGPROB_IDS);
}

#[tokio::test]
async fn generic_topk_ignores_non_letter_words() {
    let (ctx, _engine) = make_ctx(
        MockEngine::topk_only(&[
            ("apple", -0.1),
            ("a", -1.0),
            ("b", -2.0),
            ("banana", -3.0),
        ]),
        letters_spec(false),
    );
    let (_, answer, _) = score_question(
        ctx,
        Value::String("x".to_string()),
        "q".to_string(),
        choice("Pick", &["a", "b"]),
    )
    .await
    .unwrap();
    let probs = answer["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), 2);
    assert!(probs["a"].as_f64().unwrap() > probs["b"].as_f64().unwrap());
}

#[tokio::test]
async fn echo_refuses_prefix_overlapping_keys() {
    let (ctx, _engine) = make_ctx(
        MockEngine::echo(&[("x".to_string(), -1.0)]),
        letters_spec(false),
    );
    let keys = vec!["option_1".to_string(), "option_10".to_string()];
    let error = echo_scores(&ctx, "context", &keys).await.unwrap_err();
    match error {
        EngineError::Unsupported(message) => assert!(message.contains("prefix-overlapping")),
        EngineError::Other(message) => panic!("unexpected error: {message}"),
    }
}

#[tokio::test]
async fn unsupported_question_type_is_refused() {
    let (ctx, _engine) = make_ctx(MockEngine::exact(&[]), letters_spec(true));
    let error = score_question(
        ctx,
        Value::String("x".to_string()),
        "q".to_string(),
        json!({"type": "ranking", "instructions": "Rank these"}),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, EngineError::Unsupported(_)));
}

#[tokio::test]
async fn cache_preserves_criteria_order() {
    let (ctx, _engine) = make_ctx(
        MockEngine::exact(&[(97, -0.5), (98, -1.0), (99, -2.0)]),
        letters_spec(true),
    );
    let forward = json!({"q": choice("Pick", &["a", "b", "c"])});
    let reversed = json!({"q": choice("Pick", &["c", "b", "a"])});
    let first = readout_call(
        ctx.clone(),
        &Value::String("x".to_string()),
        forward.as_object().unwrap(),
    )
    .await
    .unwrap();
    let second = readout_call(
        ctx,
        &Value::String("x".to_string()),
        reversed.as_object().unwrap(),
    )
    .await
    .unwrap();
    let first_keys: Vec<&String> = first["answers"]["q"]["probabilities"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    let second_keys: Vec<&String> = second["answers"]["q"]["probabilities"]
        .as_object()
        .unwrap()
        .keys()
        .collect();
    assert_eq!(first_keys, vec!["a", "b", "c"]);
    assert_eq!(second_keys, vec!["c", "b", "a"]);
}
