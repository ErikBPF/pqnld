//! Rust ports of the readout invariants previously covered by the Python unit
//! tests. A `MockEngine` and holding `Ctx` let these run without a live engine.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use serde_json::{json, Map, Value};
use tokio::sync::{Mutex, Semaphore};

use super::engine::{BoxFuture, Capabilities, Engine, EngineError, PromptLogprobs, Surface};
use super::*;

pub(super) struct MockEngine {
    caps: Capabilities,
    tokenize_map: HashMap<String, Vec<u32>>,
    scores: HashMap<u32, f64>,
    only_known: bool,
    topk: Vec<(String, f64)>,
    prompt: PromptLogprobs,
    prompt_batch: Option<Vec<PromptLogprobs>>,
    sampled: u32,
    calls: Mutex<Vec<(Vec<u32>, String)>>,
    requests: AtomicUsize,
    gate: Option<Arc<ScoringGate>>,
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
    pub(super) fn exact(scores: &[(u32, f64)]) -> Self {
        MockEngine {
            caps: caps(true, false, false),
            tokenize_map: HashMap::new(),
            scores: scores.iter().copied().collect(),
            only_known: false,
            topk: Vec::new(),
            prompt: Vec::new(),
            prompt_batch: None,
            sampled: 0,
            calls: Mutex::new(Vec::new()),
            requests: AtomicUsize::new(0),
            gate: None,
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
            prompt_batch: None,
            sampled: 0,
            calls: Mutex::new(Vec::new()),
            requests: AtomicUsize::new(0),
            gate: None,
        }
    }

    fn echo(tokens: &[(String, f64)]) -> Self {
        MockEngine {
            caps: caps(false, false, true),
            tokenize_map: HashMap::new(),
            scores: HashMap::new(),
            only_known: false,
            topk: Vec::new(),
            prompt: tokens.iter().map(|(token, score)| (token.clone(), Some(*score))).collect(),
            prompt_batch: None,
            sampled: 0,
            calls: Mutex::new(Vec::new()),
            requests: AtomicUsize::new(0),
            gate: None,
        }
    }

    pub(super) fn request_count(&self) -> usize {
        self.requests.load(Ordering::Relaxed)
    }
}

impl Engine for MockEngine {
    fn capabilities(&self) -> Capabilities {
        self.caps
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>> {
        self.requests.fetch_add(1, Ordering::Relaxed);
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
        prompt: &'a str,
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
            self.requests.fetch_add(1, Ordering::Relaxed);
            self.calls.lock().await.push((requested, prompt.to_string()));
            if let Some(gate) = &self.gate { gate.score(prompt).await?; }
            Ok(out)
        })
    }

    fn topk<'a>(&'a self, _prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        let mut out = self.topk.clone();
        out.truncate(k);
        Box::pin(async move {
            self.requests.fetch_add(1, Ordering::Relaxed);
            Ok(out)
        })
    }

    fn prompt_logprobs<'a>(&'a self, _prompt: &'a str) -> BoxFuture<'a, Result<PromptLogprobs, EngineError>> {
        let pairs = self.prompt.clone();
        Box::pin(async move {
            self.requests.fetch_add(1, Ordering::Relaxed);
            Ok(pairs)
        })
    }

    fn prompt_logprobs_batch<'a>(&'a self, prompts: &'a [String]) -> BoxFuture<'a, Result<Vec<PromptLogprobs>, EngineError>> {
        Box::pin(async move {
            if let Some(batch) = &self.prompt_batch {
                self.requests.fetch_add(1, Ordering::Relaxed);
                return Ok(batch.clone());
            }
            let mut batch = Vec::new();
            for prompt in prompts {
                batch.push(self.prompt_logprobs(prompt).await?);
            }
            Ok(batch)
        })
    }
}

struct ScoringGate {
    active: AtomicUsize,
    maximum: AtomicUsize,
    entered: Semaphore,
    release: Semaphore,
    fail_release: Semaphore,
    held_release: Semaphore,
    finished: Semaphore,
    fail_panics: bool,
    prompts: StdMutex<Vec<String>>,
    completed: StdMutex<Vec<String>>,
}

impl ScoringGate {
    fn new(fail_panics: bool) -> Self {
        Self {
            active: AtomicUsize::new(0), maximum: AtomicUsize::new(0),
            entered: Semaphore::new(0), release: Semaphore::new(0),
            fail_release: Semaphore::new(0), fail_panics,
            held_release: Semaphore::new(0), finished: Semaphore::new(0),
            prompts: StdMutex::new(Vec::new()),
            completed: StdMutex::new(Vec::new()),
        }
    }

    async fn score(&self, prompt: &str) -> Result<(), EngineError> {
        struct Active<'a>(&'a AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) { self.0.fetch_sub(1, Ordering::Relaxed); }
        }
        let active = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        let _active = Active(&self.active);
        self.maximum.fetch_max(active, Ordering::Relaxed);
        self.prompts.lock().unwrap().push(prompt.to_string());
        self.entered.add_permits(1);
        if prompt.contains("FAIL") {
            self.fail_release.acquire().await.unwrap().forget();
            assert!(!self.fail_panics, "fixture scoring panic");
            return Err(EngineError::Unsupported("fixture scoring refusal".to_string()));
        }
        if prompt.contains("HELD") {
            self.held_release.acquire().await.unwrap().forget();
        } else {
            self.release.acquire().await.unwrap().forget();
        }
        self.completed.lock().unwrap().push(prompt.to_string());
        self.finished.add_permits(1);
        Ok(())
    }
}

fn gated_ctx(workers: usize, auto: bool, gate: Arc<ScoringGate>) -> Arc<Ctx> {
    let engine = MockEngine { sampled: 97, gate: Some(gate), ..MockEngine::exact(&[]) };
    let mut spec = letters_spec(true);
    if auto { spec.readout = "auto".to_string(); }
    let (mut ctx, _) = make_ctx(engine, spec);
    let inner = Arc::get_mut(&mut ctx).unwrap();
    inner.workers = workers;
    inner.sem = Arc::new(Semaphore::new(workers));
    ctx
}

#[tokio::test]
async fn workers_one_bounds_independent_clients_and_initial_probes() {
    use std::future::{poll_fn, Future};
    use std::task::Poll;
    for auto in [false, true] {
        let gate = Arc::new(ScoringGate::new(false));
        let ctx = gated_ctx(1, auto, gate.clone());
        let questions = json!({"q": choice("Pick", &["a", "b"])});
        let first_state = json!("first client");
        let second_state = json!("second client");
        let first = readout_call(ctx.clone(), &first_state, questions.as_object().unwrap());
        let second = readout_call(ctx, &second_state, questions.as_object().unwrap());
        tokio::pin!(first, second);
        // Poll both real clients into admission/scoring before inspecting activity.
        poll_fn(|cx| {
            assert!(first.as_mut().poll(cx).is_pending());
            assert!(second.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        }).await;
        assert_eq!(gate.maximum.load(Ordering::Relaxed), 1, "auto={auto} exceeded global budget");
        gate.release.add_permits(4);
        let (first, second) = tokio::join!(first, second);
        validate(questions.as_object().unwrap(), &first.unwrap()).unwrap();
        validate(questions.as_object().unwrap(), &second.unwrap()).unwrap();
        assert_eq!(gate.active.load(Ordering::Relaxed), 0);
        assert_eq!(gate.maximum.load(Ordering::Relaxed), 1);
    }
}

#[tokio::test]
async fn failed_concurrent_readout_cancels_and_drains_remaining_scoring() {
    for (panics, failure_first) in [(false, true), (true, true), (false, false), (true, false)] {
        let result = tokio::time::timeout(Duration::from_secs(2), async {
            let gate = Arc::new(ScoringGate::new(panics));
            let ctx = gated_ctx(2, false, gate.clone());
            let mut questions = Map::new();
            if failure_first { questions.insert("failure".to_string(), choice("FAIL", &["a", "b"])); }
            questions.insert("blocked".to_string(), choice("BLOCKED", &["a", "b"]));
            if !failure_first { questions.insert("failure".to_string(), choice("FAIL", &["a", "b"])); }
            questions.insert("queued".to_string(), choice("QUEUED", &["a", "b"]));
            let task_ctx = ctx.clone();
            let task = tokio::spawn(async move {
                readout_call(task_ctx, &Value::Null, &questions).await
            });
            gate.entered.acquire_many(2).await.unwrap().forget();
            assert_eq!(gate.active.load(Ordering::Relaxed), 2);
            gate.fail_release.add_permits(1);
            let error = task.await.unwrap().unwrap_err();
            if panics { assert!(matches!(error, EngineError::Other(_)), "{error:?}"); }
            else { assert!(matches!(error, EngineError::Unsupported(ref message) if message == "fixture scoring refusal"), "{error:?}"); }
            assert_eq!(gate.active.load(Ordering::Relaxed), 0, "detached scoring survived refusal");
            assert_eq!(ctx.sem.available_permits(), 2, "worker permits leaked");
            assert!(gate.maximum.load(Ordering::Relaxed) <= 2);
            gate.release.add_permits(1);
            let next = json!({"next": choice("NEXT", &["a", "b"])});
            let response = readout_call(ctx, &Value::Null, next.as_object().unwrap()).await.unwrap();
            validate(next.as_object().unwrap(), &response).unwrap();
            assert_eq!(gate.active.load(Ordering::Relaxed), 0);
        }).await;
        assert!(result.is_ok(), "setup, failure or recovery stalled: panics={panics}, failure_first={failure_first}");
    }
}

#[tokio::test]
async fn workers_two_preserve_answer_order_after_reverse_completion() {
    let result = tokio::time::timeout(Duration::from_secs(2), async {
        let gate = Arc::new(ScoringGate::new(false));
        let ctx = gated_ctx(2, false, gate.clone());
        let questions = json!({"z": choice("HELD", &["a", "b"]), "a": choice("FAST", &["a", "b"])});
        let task_questions = questions.clone();
        let task = tokio::spawn(async move {
            readout_call(ctx, &Value::Null, task_questions.as_object().unwrap()).await
        });
        gate.entered.acquire_many(2).await.unwrap().forget();
        assert_eq!(gate.active.load(Ordering::Relaxed), 2);
        gate.release.add_permits(1);
        gate.finished.acquire().await.unwrap().forget();
        assert!(gate.completed.lock().unwrap()[0].contains("Question:\nFAST\n"));
        gate.held_release.add_permits(1);
        let response = task.await.unwrap().unwrap();
        validate(questions.as_object().unwrap(), &response).unwrap();
        let completed = gate.completed.lock().unwrap();
        assert_eq!(completed.len(), 2);
        assert!(completed[1].contains("Question:\nHELD\n"));
        let keys: Vec<&str> = response["answers"].as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["z", "a"]);
        assert_eq!(gate.active.load(Ordering::Relaxed), 0);
        assert_eq!(gate.maximum.load(Ordering::Relaxed), 2);
    }).await;
    assert!(result.is_ok(), "reverse-completion setup or response stalled");
}

#[tokio::test]
async fn workers_one_preserves_stored_question_and_answer_order() {
    let gate = Arc::new(ScoringGate::new(false));
    gate.release.add_permits(3);
    let ctx = gated_ctx(1, false, gate.clone());
    let questions = json!({
        "z": choice("FIRST", &["a", "b"]),
        "a": choice("SECOND", &["a", "b"]),
        "m": choice("THIRD", &["a", "b"]),
    });
    let response = readout_call(ctx, &Value::Null, questions.as_object().unwrap()).await.unwrap();
    validate(questions.as_object().unwrap(), &response).unwrap();
    let prompts = gate.prompts.lock().unwrap();
    assert_eq!(prompts.len(), 3);
    for (prompt, expected) in prompts.iter().zip(["FIRST", "SECOND", "THIRD"]) {
        assert!(prompt.contains(&format!("Question:\n{expected}\n")), "{prompt}");
    }
    let keys: Vec<&str> = response["answers"].as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, vec!["z", "a", "m"]);
    assert_eq!(gate.maximum.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn workers_one_shares_budget_between_probe_and_question_scoring() {
    use std::future::{poll_fn, Future};
    use std::task::Poll;
    let gate = Arc::new(ScoringGate::new(false));
    let ctx = gated_ctx(1, false, gate.clone());
    let probing = probe(&ctx);
    let scoring = score_question(ctx.clone(), Value::Null, "q".to_string(), choice("Pick", &["a", "b"]));
    tokio::pin!(probing, scoring);
    poll_fn(|cx| {
        assert!(probing.as_mut().poll(cx).is_pending());
        assert!(scoring.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    }).await;
    assert_eq!(gate.maximum.load(Ordering::Relaxed), 1);
    gate.release.add_permits(2);
    let (probing, scoring) = tokio::join!(probing, scoring);
    assert!(matches!(probing, Ok(Mode::Lettered)));
    assert!(scoring.is_ok());
    assert_eq!(gate.active.load(Ordering::Relaxed), 0);
    assert_eq!(gate.maximum.load(Ordering::Relaxed), 1);
}

pub(super) fn make_ctx(engine: MockEngine, spec: ModelSpec) -> (Arc<Ctx>, Arc<MockEngine>) {
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

pub(super) fn malformed_questions() -> Vec<Value> {
    let oversized: Map<String, Value> = (0..256)
        .map(|i| (format!("k{i:03}"), Value::Null))
        .collect();
    vec![
        choice("Pick", &["a"]),
        json!({"type": "choice", "instructions": "Pick"}),
        choice("Pick", &[]),
        json!({"type": "choice", "instructions": "Pick", "criteria": []}),
        json!({"type": "choice", "instructions": "Pick", "criteria": oversized}),
        json!({"type": "choice", "criteria": {"a": null, "b": null}}),
        json!({"type": "noul"}),
        json!({"instructions": "Pick", "criteria": {"a": null, "b": null}}),
        json!({"type": 7, "instructions": "Pick", "criteria": {"a": null, "b": null}}),
        json!({"type": "ranking", "instructions": "Pick"}),
        Value::Null,
    ]
}

pub(super) fn structured_question_values() -> Vec<(Value, &'static str)> {
    vec![
        (json!({"task": "classify", "threshold": 0.00001}), r#"{"task":"classify","threshold":1e-05}"#),
        (json!(["label", null, 0.00001]), r#"["label",null,1e-05]"#),
        (json!(7), "7"),
        (json!(0.00001), "1e-05"),
        (json!(true), "true"),
        (Value::Null, "null"),
        (json!("plain"), "plain"),
    ]
}

#[tokio::test]
async fn structured_question_json_preserves_shared_readout_and_legacy_rendering() {
    for (value, rendered) in structured_question_values() {
        let (ctx, engine) = make_ctx(MockEngine::exact(&[(97, -0.5), (98, -1.0)]), letters_spec(true));
        let questions = json!({
            "choice": {"type": "choice", "instructions": value, "criteria": {"left": value, "right": null}},
            "binary": {"type": "noul", "instructions": value}
        });
        let result = readout_call(ctx, &json!({"context": [1, null, true]}), questions.as_object().unwrap()).await;
        assert!(result.is_ok(), "present JSON question value refused: {value}: {result:?}");
        validate(questions.as_object().unwrap(), &result.unwrap()).unwrap();
        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 2);
        for (_, prompt) in calls.iter() {
            assert!(prompt.contains(&format!("Question:\n{rendered}\n\nOptions:")), "{prompt}");
        }
        let description = if value.is_null() { "left" } else { rendered };
        assert!(calls[0].1.contains(&format!("a) {description}\nb) right\n")), "{}", calls[0].1);
    }
}

#[tokio::test]
async fn structured_question_json_succeeds_over_real_http_and_chat() {
    let result = tokio::time::timeout(Duration::from_secs(2), async {
        let (ctx, engine) = make_ctx(MockEngine::exact(&[(97, -0.5), (98, -1.0)]), letters_spec(true));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                serve_connection(stream, ctx.clone()).await;
            }
        });
        let client = reqwest::Client::builder().no_proxy().pool_max_idle_per_host(0).build().unwrap();
        let decision = json!({"state": {"context": [1, true, null]}, "questions": {
            "choice": {"type": "choice", "instructions": {"action": "route", "threshold": 0.00001},
                "criteria": {"a": {"nested": [true, null]}, "b": [1, "label"]}},
            "binary": {"type": "noul", "instructions": null}
        }});
        let response = client.post(format!("{base_url}/v1/decide")).json(&decision).send().await.unwrap();
        assert_eq!(response.status().as_u16(), 200, "structured HTTP decision refused");
        let body: Value = response.json().await.unwrap();
        validate(decision["questions"].as_object().unwrap(), &body).unwrap();
        let chat = json!({"messages": [{"role": "user", "content": decision.to_string()}]});
        let response = client.post(format!("{base_url}/v1/chat/completions")).json(&chat).send().await.unwrap();
        assert_eq!(response.status().as_u16(), 200, "structured chat decision refused");
        let body: Value = response.json().await.unwrap();
        let answer: Value = serde_json::from_str(body["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
        validate(decision["questions"].as_object().unwrap(), &answer).unwrap();
        drop(client);
        server.await.unwrap();
        let calls = engine.calls.lock().await;
        assert_eq!(calls.len(), 2, "chat should reuse the structured-question cache");
        assert!(calls[0].1.contains("Question:\n{\"action\":\"route\",\"threshold\":1e-05}\n"));
        assert!(calls[0].1.contains("a) {\"nested\":[true,null]}\nb) [1,\"label\"]\n"));
        assert!(calls[1].1.contains("Question:\nnull\n"));
    }).await;
    assert!(result.is_ok(), "loopback HTTP structured-question test stalled");
}

#[tokio::test]
async fn malformed_questions_refuse_before_any_engine_request() {
    for question in malformed_questions() {
        let (ctx, engine) = make_ctx(
            MockEngine::exact(&[]),
            ModelSpec { specific_token_scores: true, ..Default::default() },
        );
        // A valid earlier question must not be scored before a later invalid one.
        let questions = json!({"first": choice("Pick", &["a", "b"]), "bad": question});
        let result = readout_call(ctx.clone(), &Value::Null, questions.as_object().unwrap()).await;
        assert_eq!(engine.request_count(), 0, "engine accessed for {question}");
        assert!(matches!(result, Err(EngineError::Unsupported(_))), "{result:?}");
        let result = score_question(ctx, Value::Null, "bad".to_string(), question).await;
        assert_eq!(engine.request_count(), 0, "direct scoring accessed engine");
        assert!(matches!(result, Err(EngineError::Unsupported(_))), "{result:?}");
    }
    let (ctx, engine) = make_ctx(MockEngine::exact(&[]), ModelSpec::default());
    let result = readout_call(ctx, &Value::Null, &Map::new()).await;
    assert_eq!(engine.request_count(), 0, "empty questions triggered probe");
    assert!(matches!(result, Err(EngineError::Unsupported(_))));
}

#[tokio::test]
async fn http_and_chat_share_input_refusal_without_engine_access() {
    let (ctx, engine) = make_ctx(MockEngine::exact(&[]), ModelSpec::default());
    let mut cases: Vec<Value> = malformed_questions().into_iter().map(|q| json!({"q": q})).collect();
    cases.push(json!({}));
    for questions in cases {
        let decision = json!({"state": null, "questions": questions});
        let response = handle_decide(ctx.clone(), Bytes::from(decision.to_string())).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY, "{decision}");
        let chat = json!({"messages": [{"role": "user", "content": decision.to_string()}]});
        let response = handle_chat(ctx.clone(), Bytes::from(chat.to_string())).await;
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY, "{decision}");
        assert_eq!(engine.request_count(), 0, "malformed transport request reached engine");
    }
    let response = handle_decide(ctx, Bytes::from_static(b"{}" )).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn valid_http_and_chat_retain_typed_envelopes() {
    let (ctx, engine) = make_ctx(MockEngine::exact(&[(97, -0.5), (98, -1.0)]), letters_spec(true));
    let decision = json!({"state": {
        "meeting": {"days": ["Tuesday", "Friday"], "urgent": true},
        "attendees": 3, "metadata": [null, {"confidence": 0.5}], "note": "café"
    }, "model": "alias", "questions": {
        "choice": choice("", &["a", "b"]), "binary": {"type": "noul", "instructions": ""}
    }});
    let response = handle_decide(ctx.clone(), Bytes::from(decision.to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["model"], "alias");
    validate(decision["questions"].as_object().unwrap(), &body).unwrap();
    let chat = json!({"messages": [{"role": "user", "content": decision.to_string()}]});
    let response = handle_chat(ctx, Bytes::from(chat.to_string())).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let decision_body: Value = serde_json::from_str(body["choices"][0]["message"]["content"].as_str().unwrap()).unwrap();
    validate(decision["questions"].as_object().unwrap(), &decision_body).unwrap();
    let calls = engine.calls.lock().await;
    assert_eq!(calls.len(), 2, "chat should reuse the structured-state decision cache");
    for (_, prompt) in calls.iter() {
        assert!(prompt.contains("\"metadata\":[null,{\"confidence\":0.5}]"), "{prompt}");
        assert!(prompt.contains("\"days\":[\"Tuesday\",\"Friday\"]"), "{prompt}");
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
    // Independent test-owned labels/IDs: no production alphabet or mapping oracle.
    let labels: Vec<String> = (0..count).map(|i| char::from_u32(0x1000 + i as u32).unwrap().to_string()).collect();
    let scores: Vec<(u32, f64)> = (0..count).map(|i| {
        (5000 + i as u32, 2.0 * (((i + 1) as f64) / 151.0).ln())
    }).collect();
    let mut fixture = MockEngine::exact_strict(&scores);
    fixture.tokenize_map = labels.iter().enumerate().map(|(i, label)| {
        (label.clone(), vec![5000 + i as u32])
    }).collect();
    let mut spec = letters_spec(true);
    spec.letters = labels.concat();
    spec.temperature = 2.0;
    let (ctx, engine) = make_ctx(fixture, spec);
    let (_, answer, _) = score_question(ctx, Value::String("x".to_string()), "q".to_string(), question)
        .await
        .unwrap();
    let probs = answer["probabilities"].as_object().unwrap();
    assert_eq!(probs.len(), count);
    // Scores are 2*ln((i+1)/151), T=2: P(i)=(i+1)/sum(1..151).
    // Closed-form oracle never reads returned scores or production normalization.
    for (i, key) in keys.iter().enumerate() {
        let actual = probs[key].as_f64().unwrap();
        let expected = (i + 1) as f64 / 11476.0;
        assert!((actual - expected).abs() < 1e-12, "{key}: expected {expected}, got {actual}");
    }
    assert_eq!(answer["choice"], "k150", "winner must come from the second chunk");
    let calls = engine.calls.lock().await;
    assert_eq!(calls.len(), 2, "expected two id chunks");
    assert_eq!(calls[0].0, (5000..5128).collect::<Vec<u32>>());
    assert_eq!(calls[1].0, (5128..5151).collect::<Vec<u32>>());
    assert_eq!(calls[0].1, calls[1].1, "chunks must score the same conditioned prompt");
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
async fn echo_refuses_empty_or_unextended_evidence() {
    for prompt in [vec![], vec![("context".to_string(), -1.0)]] {
        let (ctx, _) = make_ctx(MockEngine::echo(&prompt), ModelSpec {
            readout: "echo".to_string(), ..Default::default()
        });
        let keys = vec!["a".to_string(), "b".to_string()];
        let result = echo_scores(&ctx, "context", &keys).await;
        assert!(matches!(result, Err(EngineError::Other(_))), "empty suffix accepted: {result:?}");
    }
}

#[tokio::test]
async fn echo_refuses_wrong_batch_cardinality() {
    for count in [0, 1, 3] {
        let engine = MockEngine {
            prompt: vec![("ctx".to_string(), None)],
            prompt_batch: Some(vec![vec![("ctx".to_string(), None), ("suffix".to_string(), Some(-1.0))]; count]),
            ..MockEngine::echo(&[])
        };
        let (ctx, _) = make_ctx(engine, ModelSpec::default());
        let result = echo_scores(&ctx, "ctx", &["a".to_string(), "b".to_string()]).await;
        assert!(matches!(result, Err(EngineError::Other(_))), "accepted {count} scores for 2 options: {result:?}");
    }
}

#[tokio::test]
async fn echo_refuses_nonfinite_scores_and_sum_overflow() {
    let context = vec![("ctx".to_string(), None)];
    let suffix = vec![("ctx".to_string(), None), ("a".to_string(), Some(-1.0))];
    let mut cases = Vec::new();
    for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.push((vec![("ctx".to_string(), Some(score))], suffix.clone()));
        cases.push((context.clone(), vec![("ctx".to_string(), None), ("a".to_string(), Some(score))]));
    }
    cases.push((context.clone(), vec![("ctx".to_string(), None), ("a".to_string(), Some(-f64::MAX)), ("b".to_string(), Some(-f64::MAX))]));
    cases.push((context.clone(), vec![("ctx".to_string(), None), ("a".to_string(), None)]));
    for (prompt, pairs) in cases {
        let engine = MockEngine { prompt, prompt_batch: Some(vec![pairs.clone(), pairs]), ..MockEngine::echo(&[]) };
        let (ctx, _) = make_ctx(engine, ModelSpec::default());
        let result = echo_scores(&ctx, "ctx", &["a".to_string(), "b".to_string()]).await;
        assert!(matches!(result, Err(EngineError::Other(_))), "accepted unusable scores: {result:?}");
    }
}

#[tokio::test]
async fn echo_preserves_null_prefix_and_boundary_retokenization() {
    let engine = MockEngine {
        prompt: vec![("ctx".to_string(), None), ("boundary".to_string(), Some(-1.0))],
        prompt_batch: Some(vec![
            vec![("ctx".to_string(), None), ("merged_a".to_string(), Some(-2.0))],
            vec![("ctx".to_string(), None), ("boundary".to_string(), Some(-1.0)), ("b".to_string(), Some(-1.0))],
        ]),
        ..MockEngine::echo(&[])
    };
    let (ctx, _) = make_ctx(engine, ModelSpec { readout: "echo".to_string(), ..Default::default() });
    let questions = json!({"q": choice("Pick", &["a", "b"])});
    let response = readout_call(ctx, &Value::Null, questions.as_object().unwrap()).await.unwrap();
    validate(questions.as_object().unwrap(), &response).unwrap();
    assert_eq!(response["answers"]["q"]["choice"], "b");
    let probability = response["answers"]["q"]["probabilities"]["b"].as_f64().unwrap();
    assert!((probability - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-12);
}

#[test]
fn input_validator_accepts_255_choices_without_promising_tokenizer_capacity() {
    let criteria: Map<String, Value> = (0..255).map(|i| (format!("k{i:03}"), Value::Null)).collect();
    validate_question_input("q", &json!({"type": "choice", "instructions": "", "criteria": criteria})).unwrap();
}

#[tokio::test]
async fn unusable_echo_evidence_returns_http_error_not_probabilities() {
    let (ctx, _) = make_ctx(MockEngine::echo(&[]), ModelSpec { readout: "echo".to_string(), ..Default::default() });
    let decision = json!({"state": null, "questions": {"q": choice("Pick", &["a", "b"])}});
    let response = handle_decide(ctx, Bytes::from(decision.to_string())).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value = serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert!(body.get("error").is_some());
    assert!(body.get("answers").is_none());
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
