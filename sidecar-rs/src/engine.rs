// Minimal multi-engine abstraction for the pqnld readout.
//
// The readout needs three things from an inference server: tokenization, a
// scored answer slot (either caller-supplied token ids or generic top-k), and
// prompt/continuation logprobs for the echo fallback. Each server exposes those
// differently (see docs/engine-capabilities.md); the adapters here keep that
// per-engine JSON out of main.rs.
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Debug)]
pub enum EngineError {
    Unsupported(String),
    Other(String),
}

const CAPACITY_MARKERS: [&str; 9] = [
    "options per choice",
    "a choice needs at least two options",
    "a score takes 2 to 10 levels",
    "the canvas holds",
    "maximum context length",
    "maximum model length",
    "longer than the maximum model length",
    "context window",
    "too many tokens",
];

/// Where a server exposes answer-slot logprobs.
#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Surface {
    ChatCompletions,
    Completions,
    NativeGenerate,
    Score,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    pub tokenize: bool,
    pub exact_ids: bool,
    pub top_k: bool,
    pub prompt_logprobs: bool,
    pub max_top_k: Option<usize>,
    pub max_explicit_ids: Option<usize>,
    pub surface: Surface,
}

/// One answer-slot query. All methods perform the engine HTTP call and parse
/// the response; `Err(Unsupported)` is the 422 path, `Err(Other)` the 500 path.
///
/// For `score_token_ids` the first returned pair is the sampled answer token
/// (when the engine reports it), so the caller can recover the top token the
/// same way the vLLM path always has.
pub trait Engine: Send + Sync {
    fn capabilities(&self) -> Capabilities;
    fn tokenize<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>>;
    fn score_token_ids<'a>(
        &'a self,
        prompt: &'a str,
        ids: &'a [u32],
    ) -> BoxFuture<'a, Result<Vec<(u32, f64)>, EngineError>>;
    fn topk<'a>(&'a self, prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>>;
    fn prompt_logprobs<'a>(&'a self, raw_prompt: &'a str) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>>;

    /// Batched echo; the vLLM path sends the prompt array in one request, which
    /// is what bounds prompt-logprob memory. Other engines loop.
    fn prompt_logprobs_batch<'a>(
        &'a self,
        prompts: &'a [String],
    ) -> BoxFuture<'a, Result<Vec<Vec<(String, f64)>>, EngineError>> {
        Box::pin(async move {
            let mut out = Vec::with_capacity(prompts.len());
            for prompt in prompts {
                out.push(self.prompt_logprobs(prompt).await?);
            }
            Ok(out)
        })
    }

    /// Prompt tokens reported by the most recent response, for the wire `usage`.
    fn last_usage(&self) -> i64 {
        0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EngineKind {
    Vllm,
    LlamaCpp,
    Sglang,
    OpenAi,
}

pub fn parse_kind(value: &str) -> Option<EngineKind> {
    match value {
        "vllm" => Some(EngineKind::Vllm),
        "llamacpp" | "llama.cpp" | "llama" => Some(EngineKind::LlamaCpp),
        "sglang" => Some(EngineKind::Sglang),
        "openai" => Some(EngineKind::OpenAi),
        _ => None,
    }
}

/// Best-effort startup probe used by `--engine-kind auto`.
pub async fn detect_kind(client: &reqwest::Client, base_url: &str) -> EngineKind {
    if let Some(value) = get_json(client, base_url, "/openapi.json").await {
        let text = value.to_string().to_lowercase();
        if text.contains("sglang") {
            return EngineKind::Sglang;
        }
        if text.contains("llama.cpp") || text.contains("llama-server") {
            return EngineKind::LlamaCpp;
        }
        if text.contains("vllm") {
            return EngineKind::Vllm;
        }
    }
    if let Some(value) = get_json(client, base_url, "/v1/models").await {
        let text = value.to_string().to_lowercase();
        if text.contains("sglang") {
            return EngineKind::Sglang;
        }
        if text.contains("llama") {
            return EngineKind::LlamaCpp;
        }
    }
    EngineKind::Vllm
}

pub fn build(
    kind: EngineKind,
    client: reqwest::Client,
    base_url: String,
    model: String,
    system_prompt: String,
    enable_thinking: bool,
) -> Arc<dyn Engine> {
    let endpoint = Endpoint {
        client,
        base_url,
        model,
        system_prompt,
        enable_thinking,
        usage: AtomicI64::new(0),
    };
    match kind {
        EngineKind::Vllm => Arc::new(VllmEngine { endpoint }),
        EngineKind::LlamaCpp => Arc::new(LlamaCppEngine { endpoint }),
        EngineKind::Sglang => Arc::new(SglangEngine { endpoint }),
        EngineKind::OpenAi => Arc::new(OpenAiEngine { endpoint }),
    }
}

async fn get_json(client: &reqwest::Client, base_url: &str, path: &str) -> Option<Value> {
    let url = format!("{base_url}{path}");
    let response = client.get(&url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json::<Value>().await.ok()
}

struct Endpoint {
    client: reqwest::Client,
    base_url: String,
    model: String,
    system_prompt: String,
    enable_thinking: bool,
    usage: AtomicI64,
}

impl Endpoint {
    async fn post_json(&self, path: &str, body: &Value) -> Result<Value, EngineError> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| EngineError::Other(format!("engine request failed: {e}")))?;
        let status = response.status();
        let body_text = response
            .text()
            .await
            .map_err(|e| EngineError::Other(format!("engine read failed: {e}")))?;
        if !status.is_success() {
            let capacity = matches!(status.as_u16(), 400 | 413 | 422)
                && CAPACITY_MARKERS.iter().any(|m| body_text.contains(m));
            return Err(if capacity {
                EngineError::Unsupported(body_text)
            } else {
                EngineError::Other(format!("engine HTTP {}: {}", status.as_u16(), body_text))
            });
        }
        let value: Value = serde_json::from_str(&body_text)
            .map_err(|e| EngineError::Other(format!("engine JSON parse failed: {e}")))?;
        if let Some(tokens) = value
            .get("usage")
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|t| t.as_i64())
        {
            self.usage.store(tokens, Ordering::Relaxed);
        }
        Ok(value)
    }

    fn chat_body(&self, prompt: &str, extra: Value) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": true,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        });
        if let Value::Object(extra) = extra {
            for (key, value) in extra {
                body[key] = value;
            }
        }
        body
    }

    fn echo_body(&self, prompt: Value, logprobs: i64) -> Value {
        json!({
            "model": self.model,
            "prompt": prompt,
            "echo": true,
            "max_tokens": 0,
            "temperature": 0,
            "logprobs": logprobs,
        })
    }
}

// --- vLLM: exact ids (logprob_token_ids) + generic top-k + /tokenize + echo ---

pub struct VllmEngine {
    endpoint: Endpoint,
}

impl Engine for VllmEngine {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tokenize: true,
            exact_ids: true,
            top_k: true,
            prompt_logprobs: true,
            max_top_k: Some(20),
            max_explicit_ids: Some(128),
            surface: Surface::ChatCompletions,
        }
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>> {
        Box::pin(async move {
            let body = json!({
                "model": self.endpoint.model,
                "prompt": text,
                "add_special_tokens": false,
            });
            let payload = self.endpoint.post_json("/tokenize", &body).await?;
            Ok(parse_token_ids(&payload))
        })
    }

    fn score_token_ids<'a>(
        &'a self,
        prompt: &'a str,
        ids: &'a [u32],
    ) -> BoxFuture<'a, Result<Vec<(u32, f64)>, EngineError>> {
        Box::pin(async move {
            let request = self.endpoint.chat_body(
                prompt,
                json!({
                    "top_logprobs": 0,
                    "logprob_token_ids": ids,
                    "return_tokens_as_token_ids": true,
                }),
            );
            let payload = self.endpoint.post_json("/v1/chat/completions", &request).await?;
            parse_vllm_exact(&payload).ok_or_else(no_answer_slot)
        })
    }

    fn topk<'a>(&'a self, prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let request = self.endpoint.chat_body(prompt, json!({"top_logprobs": k}));
            let payload = self.endpoint.post_json("/v1/chat/completions", &request).await?;
            parse_chat_topk(&payload).ok_or_else(no_answer_slot)
        })
    }

    fn prompt_logprobs<'a>(&'a self, raw_prompt: &'a str) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let body = self.endpoint.echo_body(json!(raw_prompt), 1);
            let payload = self.endpoint.post_json("/v1/completions", &body).await?;
            Ok(payload
                .get("choices")
                .and_then(|c| c.get(0))
                .map(parse_echo_choice)
                .unwrap_or_default())
        })
    }

    fn prompt_logprobs_batch<'a>(
        &'a self,
        prompts: &'a [String],
    ) -> BoxFuture<'a, Result<Vec<Vec<(String, f64)>>, EngineError>> {
        Box::pin(async move {
            let body = self.endpoint.echo_body(json!(prompts), 1);
            let payload = self.endpoint.post_json("/v1/completions", &body).await?;
            Ok(payload
                .get("choices")
                .and_then(|c| c.as_array())
                .map(|choices| choices.iter().map(parse_echo_choice).collect())
                .unwrap_or_default())
        })
    }

    fn last_usage(&self) -> i64 {
        self.endpoint.usage.load(Ordering::Relaxed)
    }
}

// --- llama.cpp: top-k only (native /completion n_probs) + /tokenize ---

pub struct LlamaCppEngine {
    endpoint: Endpoint,
}

impl Engine for LlamaCppEngine {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tokenize: true,
            exact_ids: false,
            top_k: true,
            prompt_logprobs: false,
            max_top_k: None,
            max_explicit_ids: None,
            surface: Surface::NativeGenerate,
        }
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>> {
        Box::pin(async move {
            let body = json!({"content": text, "add_special": false});
            let payload = self.endpoint.post_json("/tokenize", &body).await?;
            Ok(parse_token_ids(&payload))
        })
    }

    fn score_token_ids<'a>(
        &'a self,
        _prompt: &'a str,
        _ids: &'a [u32],
    ) -> BoxFuture<'a, Result<Vec<(u32, f64)>, EngineError>> {
        Box::pin(async move {
            Err(EngineError::Unsupported(
                "llama.cpp has no exact-token-id scoring".to_string(),
            ))
        })
    }

    fn topk<'a>(&'a self, prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let body = json!({
                "model": self.endpoint.model,
                "prompt": prompt,
                "n_predict": 1,
                "temperature": 0,
                "n_probs": k,
            });
            let payload = self.endpoint.post_json("/completion", &body).await?;
            parse_llamacpp_topk(&payload).ok_or_else(no_answer_slot)
        })
    }

    fn prompt_logprobs<'a>(&'a self, _raw_prompt: &'a str) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            Err(EngineError::Unsupported(
                "llama.cpp native completion has no echo prompt logprobs".to_string(),
            ))
        })
    }

    fn last_usage(&self) -> i64 {
        self.endpoint.usage.load(Ordering::Relaxed)
    }
}

// --- SGLang: native exact ids via /generate meta_info ---

pub struct SglangEngine {
    endpoint: Endpoint,
}

impl SglangEngine {
    async fn generate(&self, body: &Value) -> Result<Value, EngineError> {
        self.endpoint.post_json("/generate", body).await
    }
}

impl Engine for SglangEngine {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tokenize: true,
            exact_ids: true,
            top_k: true,
            prompt_logprobs: true,
            max_top_k: None,
            max_explicit_ids: None,
            surface: Surface::NativeGenerate,
        }
    }

    fn tokenize<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>> {
        Box::pin(async move {
            let body = json!({"text": text, "add_special_tokens": false});
            let payload = self.endpoint.post_json("/tokenize", &body).await?;
            Ok(parse_token_ids(&payload))
        })
    }

    fn score_token_ids<'a>(
        &'a self,
        prompt: &'a str,
        ids: &'a [u32],
    ) -> BoxFuture<'a, Result<Vec<(u32, f64)>, EngineError>> {
        Box::pin(async move {
            let body = json!({
                "text": prompt,
                "return_logprob": true,
                "logprob_start_len": 0,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0,
                    "token_ids_logprob": ids,
                },
            });
            let payload = self.generate(&body).await?;
            parse_sglang_ids(&payload).ok_or_else(no_answer_slot)
        })
    }

    fn topk<'a>(&'a self, prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let body = json!({
                "text": prompt,
                "return_logprob": true,
                "logprob_start_len": 0,
                "sampling_params": {
                    "max_new_tokens": 1,
                    "temperature": 0,
                    "top_logprobs_num": k,
                },
            });
            let payload = self.generate(&body).await?;
            parse_sglang_topk(&payload).ok_or_else(no_answer_slot)
        })
    }

    fn prompt_logprobs<'a>(&'a self, raw_prompt: &'a str) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let body = json!({
                "text": raw_prompt,
                "return_logprob": true,
                "logprob_start_len": 0,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            });
            let payload = self.generate(&body).await?;
            Ok(parse_sglang_prompt(&payload))
        })
    }

    fn last_usage(&self) -> i64 {
        self.endpoint.usage.load(Ordering::Relaxed)
    }
}

// --- OpenAI: top-k chat only; no tokenize, no exact ids ---

pub struct OpenAiEngine {
    endpoint: Endpoint,
}

impl Engine for OpenAiEngine {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            tokenize: false,
            exact_ids: false,
            top_k: true,
            prompt_logprobs: true,
            max_top_k: Some(20),
            max_explicit_ids: None,
            surface: Surface::ChatCompletions,
        }
    }

    fn tokenize<'a>(&'a self, _text: &'a str) -> BoxFuture<'a, Result<Vec<u32>, EngineError>> {
        Box::pin(async move {
            Err(EngineError::Unsupported(
                "the OpenAI API has no tokenize endpoint".to_string(),
            ))
        })
    }

    fn score_token_ids<'a>(
        &'a self,
        _prompt: &'a str,
        _ids: &'a [u32],
    ) -> BoxFuture<'a, Result<Vec<(u32, f64)>, EngineError>> {
        Box::pin(async move {
            Err(EngineError::Unsupported(
                "the OpenAI API has no exact-token-id scoring".to_string(),
            ))
        })
    }

    fn topk<'a>(&'a self, prompt: &'a str, k: usize) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let request = self
                .endpoint
                .chat_body(prompt, json!({"top_logprobs": k.min(20)}));
            let payload = self.endpoint.post_json("/v1/chat/completions", &request).await?;
            parse_chat_topk(&payload).ok_or_else(no_answer_slot)
        })
    }

    fn prompt_logprobs<'a>(&'a self, raw_prompt: &'a str) -> BoxFuture<'a, Result<Vec<(String, f64)>, EngineError>> {
        Box::pin(async move {
            let body = self.endpoint.echo_body(json!(raw_prompt), 5);
            let payload = self.endpoint.post_json("/v1/completions", &body).await?;
            Ok(payload
                .get("choices")
                .and_then(|c| c.get(0))
                .map(parse_echo_choice)
                .unwrap_or_default())
        })
    }

    fn last_usage(&self) -> i64 {
        self.endpoint.usage.load(Ordering::Relaxed)
    }
}

fn no_answer_slot() -> EngineError {
    EngineError::Other("chat response carried no logprobs at the answer slot".to_string())
}

fn parse_token_ids(payload: &Value) -> Vec<u32> {
    payload
        .get("tokens")
        .and_then(|t| t.as_array())
        .map(|tokens| {
            tokens
                .iter()
                .filter_map(|token| {
                    token
                        .as_u64()
                        .or_else(|| token.get("id").and_then(|id| id.as_u64()))
                        .map(|id| id as u32)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn token_id_of(token: &str) -> Option<u32> {
    token
        .strip_prefix("token_id:")
        .and_then(|id| id.parse::<u32>().ok())
}

fn parse_vllm_exact(payload: &Value) -> Option<Vec<(u32, f64)>> {
    let content = payload
        .get("choices")?
        .get(0)?
        .get("logprobs")?
        .get("content")?
        .as_array()?;
    let first = content.first()?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    if let (Some(token), Some(logprob)) = (
        first.get("token").and_then(|t| t.as_str()),
        first.get("logprob").and_then(|l| l.as_f64()),
    ) {
        if let Some(id) = token_id_of(token) {
            if seen.insert(id) {
                out.push((id, logprob));
            }
        }
    }
    if let Some(tops) = first.get("top_logprobs").and_then(|t| t.as_array()) {
        for item in tops {
            if let (Some(token), Some(logprob)) = (
                item.get("token").and_then(|t| t.as_str()),
                item.get("logprob").and_then(|l| l.as_f64()),
            ) {
                if let Some(id) = token_id_of(token) {
                    if seen.insert(id) {
                        out.push((id, logprob));
                    }
                }
            }
        }
    }
    Some(out)
}

fn parse_chat_topk(payload: &Value) -> Option<Vec<(String, f64)>> {
    let content = payload
        .get("choices")?
        .get(0)?
        .get("logprobs")?
        .get("content")?
        .as_array()?;
    let first = content.first()?;
    let mut out = Vec::new();
    if let Some(tops) = first.get("top_logprobs").and_then(|t| t.as_array()) {
        for item in tops {
            if let (Some(token), Some(logprob)) = (
                item.get("token").and_then(|t| t.as_str()),
                item.get("logprob").and_then(|l| l.as_f64()),
            ) {
                out.push((token.to_string(), logprob));
            }
        }
    }
    Some(out)
}

fn parse_llamacpp_topk(payload: &Value) -> Option<Vec<(String, f64)>> {
    let first = payload.get("completion_probabilities")?.as_array()?.first()?;
    if let Some(tops) = first.get("top_logprobs").and_then(|t| t.as_array()) {
        let mut out = Vec::new();
        for item in tops {
            if let (Some(token), Some(logprob)) = (
                item.get("token").and_then(|t| t.as_str()),
                item.get("logprob").and_then(|l| l.as_f64()),
            ) {
                out.push((token.to_string(), logprob));
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    let token = first.get("token").and_then(|t| t.as_str())?;
    let logprob = first.get("logprob").and_then(|l| l.as_f64())?;
    Some(vec![(token.to_string(), logprob)])
}

fn parse_sglang_ids(payload: &Value) -> Option<Vec<(u32, f64)>> {
    let meta = payload.get("meta_info")?;
    let mut out = Vec::new();
    if let Some(items) = meta
        .get("output_token_ids_logprobs")
        .and_then(|v| v.as_array())
        .and_then(|tokens| tokens.first())
        .and_then(|v| v.as_array())
    {
        for item in items {
            if let Some(pair) = item.as_array() {
                if let (Some(logprob), Some(id)) = (
                    pair.first().and_then(|v| v.as_f64()),
                    pair.get(1).and_then(|v| v.as_u64()),
                ) {
                    out.push((id as u32, logprob));
                }
            }
        }
    }
    if out.is_empty() {
        if let Some(item) = meta
            .get("output_token_logprobs")
            .and_then(|v| v.as_array())
            .and_then(|tokens| tokens.first())
            .and_then(|v| v.as_array())
        {
            if let (Some(logprob), Some(id)) = (
                item.first().and_then(|v| v.as_f64()),
                item.get(1).and_then(|v| v.as_u64()),
            ) {
                out.push((id as u32, logprob));
            }
        }
    }
    Some(out)
}

fn parse_sglang_topk(payload: &Value) -> Option<Vec<(String, f64)>> {
    let meta = payload.get("meta_info")?;
    let mut out = Vec::new();
    if let Some(items) = meta
        .get("output_top_logprobs")
        .and_then(|v| v.as_array())
        .and_then(|tokens| tokens.first())
        .and_then(|v| v.as_array())
    {
        for item in items {
            if let Some(triple) = item.as_array() {
                if let (Some(logprob), Some(token)) = (
                    triple.first().and_then(|v| v.as_f64()),
                    triple.get(2).and_then(|v| v.as_str()),
                ) {
                    out.push((token.to_string(), logprob));
                }
            }
        }
    }
    if out.is_empty() {
        if let Some(item) = meta
            .get("output_token_logprobs")
            .and_then(|v| v.as_array())
            .and_then(|tokens| tokens.first())
            .and_then(|v| v.as_array())
        {
            if let (Some(logprob), Some(token)) = (
                item.first().and_then(|v| v.as_f64()),
                item.get(2).and_then(|v| v.as_str()),
            ) {
                out.push((token.to_string(), logprob));
            }
        }
    }
    Some(out)
}

fn parse_sglang_prompt(payload: &Value) -> Vec<(String, f64)> {
    payload
        .get("meta_info")
        .and_then(|meta| meta.get("input_token_logprobs"))
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let triple = item.as_array()?;
                    let logprob = triple.first()?.as_f64()?;
                    let token = triple.get(2)?.as_str()?;
                    Some((token.to_string(), logprob))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_echo_choice(choice: &Value) -> Vec<(String, f64)> {
    let logprobs = match choice.get("logprobs") {
        Some(logprobs) => logprobs,
        None => return Vec::new(),
    };
    match (
        logprobs.get("tokens").and_then(|t| t.as_array()),
        logprobs.get("token_logprobs").and_then(|t| t.as_array()),
    ) {
        (Some(tokens), Some(token_logprobs)) => tokens
            .iter()
            .zip(token_logprobs.iter())
            .filter_map(|(token, logprob)| {
                Some((token.as_str()?.to_string(), logprob.as_f64()?))
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vllm_exact_id_response() {
        let payload = json!({
            "choices": [{"logprobs": {"content": [{
                "token": "token_id:66",
                "logprob": -0.01,
                "top_logprobs": [
                    {"token": "token_id:66", "logprob": -0.01},
                    {"token": "token_id:64", "logprob": -3.2},
                    {"token": "token_id:65", "logprob": -4.0}
                ]
            }]}}]
        });
        assert_eq!(
            parse_vllm_exact(&payload).unwrap(),
            vec![(66, -0.01), (64, -3.2), (65, -4.0)]
        );
    }

    #[test]
    fn vllm_exact_id_request_shape() {
        let endpoint = test_endpoint();
        let request = endpoint.chat_body(
            "body",
            json!({"top_logprobs": 0, "logprob_token_ids": [64, 65, 66], "return_tokens_as_token_ids": true}),
        );
        assert_eq!(request["top_logprobs"], json!(0));
        assert_eq!(request["logprob_token_ids"], json!([64, 65, 66]));
        assert_eq!(request["return_tokens_as_token_ids"], json!(true));
        assert_eq!(request["logprobs"], json!(true));
        assert_eq!(request["messages"][0]["content"], json!("system"));
        assert_eq!(request["messages"][1]["content"], json!("body"));
    }

    #[test]
    fn vllm_generic_topk_response() {
        let payload = json!({
            "choices": [{"logprobs": {"content": [{
                "token": "b",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "b", "logprob": -0.1},
                    {"token": "a", "logprob": -2.0}
                ]
            }]}}]
        });
        assert_eq!(
            parse_chat_topk(&payload).unwrap(),
            vec![("b".to_string(), -0.1), ("a".to_string(), -2.0)]
        );
    }

    #[test]
    fn llamacpp_n_probs_response() {
        let payload = json!({
            "content": "b",
            "completion_probabilities": [{
                "id": 66,
                "token": "b",
                "logprob": -0.1,
                "top_logprobs": [
                    {"id": 66, "token": "b", "logprob": -0.1},
                    {"id": 64, "token": "a", "logprob": -2.5}
                ]
            }]
        });
        assert_eq!(
            parse_llamacpp_topk(&payload).unwrap(),
            vec![("b".to_string(), -0.1), ("a".to_string(), -2.5)]
        );
    }

    #[test]
    fn llamacpp_n_probs_request_shape() {
        let endpoint = test_endpoint();
        let request = json!({
            "model": endpoint.model,
            "prompt": "body",
            "n_predict": 1,
            "temperature": 0,
            "n_probs": 32,
        });
        assert_eq!(request["n_probs"], json!(32));
    }

    #[test]
    fn sglang_native_response() {
        let payload = json!({
            "meta_info": {
                "output_token_logprobs": [[-0.2, 66, "b"]],
                "output_top_logprobs": [[[-0.2, 66, "b"], [-2.0, 64, "a"]]],
                "output_token_ids_logprobs": [[[-0.2, 66], [-3.0, 64]]]
            }
        });
        assert_eq!(
            parse_sglang_ids(&payload).unwrap(),
            vec![(66, -0.2), (64, -3.0)]
        );
        assert_eq!(
            parse_sglang_topk(&payload).unwrap(),
            vec![("b".to_string(), -0.2), ("a".to_string(), -2.0)]
        );
    }

    #[test]
    fn openai_top_logprobs_response() {
        let payload = json!({
            "choices": [{"logprobs": {"content": [{
                "token": "b",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "b", "logprob": -0.1},
                    {"token": "a", "logprob": -2.0}
                ]
            }], "refusal": null}}]
        });
        assert_eq!(
            parse_chat_topk(&payload).unwrap(),
            vec![("b".to_string(), -0.1), ("a".to_string(), -2.0)]
        );
    }

    #[test]
    fn echo_logprobs_response() {
        let choice = json!({
            "logprobs": {
                "tokens": ["a", "b"],
                "token_logprobs": [-0.1, -0.2],
                "top_logprobs": [null, null]
            }
        });
        assert_eq!(
            parse_echo_choice(&choice),
            vec![("a".to_string(), -0.1), ("b".to_string(), -0.2)]
        );
    }

    fn test_endpoint() -> Endpoint {
        Endpoint {
            client: reqwest::Client::new(),
            base_url: "http://127.0.0.1:11542".to_string(),
            model: "test-model".to_string(),
            system_prompt: "system".to_string(),
            enable_thinking: false,
            usage: AtomicI64::new(0),
        }
    }
}
