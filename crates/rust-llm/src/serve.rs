//! The HTTP surface: two OpenAI-shaped endpoints, hand-written over
//! [`std::net::TcpListener`].
//!
//! No HTTP crate and no async runtime. There are two routes, the bodies are
//! JSON that `serde_json` already parses, and the forward pass is blocking and
//! uses every core, so there is nothing for an executor to interleave and a
//! framework would be a large dependency tree against the no-C-or-C++ rule.
//! `docs/decisions.md` has the whole argument.
//!
//! **One request at a time.** The [`Runner`] and its KV cache are the
//! generation state and there is one of each, so the accept loop is
//! single-threaded: a second client waits in the listener's backlog rather than
//! being refused, which is also all the hardware can do, since one token
//! already saturates the cores. The cost is that a connection holds the slot
//! while it is being read, which is what the read timeout is for.
//!
//! The command line's values are the per-request defaults ([`Defaults`]); a
//! request overrides the fields it sends and keeps the rest.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::str::FromStr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::chat::{self, ChatOptions, Message, Role};
use crate::forward::Runner;
use crate::generate::{self, Stop, Timings};
use crate::sample::{Sampler, Sampling, entropy_seed};
use crate::tokenizer::Tokenizer;

/// How long a connection may go without sending anything, and how long a write
/// to it may block. A client that connects and then says nothing, or stops
/// reading half-way through a stream, would otherwise hold the one generation
/// slot for ever.
const TIMEOUT: Duration = Duration::from_secs(30);
/// The request line and the headers together.
const MAX_HEADER_BYTES: u64 = 16 * 1024;
/// A body larger than this is refused without reading it. The context holds a
/// few thousand tokens, so a megabyte of JSON is already far more prompt than
/// the model can be given.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// The per-request defaults, which are the values the command line was given.
pub struct Defaults {
    /// What the answers report as `model`: the file that is loaded, not
    /// whatever the request asked for, because this is what answered.
    pub model: String,
    /// `--system`, used by a chat request that sends no system message itself.
    pub system: Option<String>,
    pub max_tokens: usize,
    pub thinking: bool,
    pub sampling: Sampling,
    /// `--seed`, if it was given: then every request without a seed of its own
    /// replays that one. Without it each request draws a fresh seed, which is
    /// logged so that request can be replayed.
    pub seed: Option<u64>,
}

/// Serves until the process is killed.
///
/// Every way one request can go wrong ends with that request: a bad body is a
/// 400, a client that went away is simply gone, and neither takes the server
/// down.
pub fn serve(
    listener: &TcpListener,
    runner: &mut Runner<'_>,
    tokenizer: &Tokenizer,
    vocab: usize,
    defaults: &Defaults,
) -> std::io::Result<()> {
    let mut server = Server {
        runner,
        tokenizer,
        vocab,
        defaults,
    };
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            // One failed accept is not the end of the server.
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        let _ = stream.set_read_timeout(Some(TIMEOUT));
        let _ = stream.set_write_timeout(Some(TIMEOUT));
        if let Err(e) = server.handle(&mut stream) {
            eprintln!("could not answer the request: {e}");
        }
    }
    Ok(())
}

/// What the server owns while it runs. One instance, which is what makes
/// generation serialized without a lock anywhere.
struct Server<'a, 'm> {
    runner: &'a mut Runner<'m>,
    tokenizer: &'a Tokenizer,
    vocab: usize,
    defaults: &'a Defaults,
}

impl Server<'_, '_> {
    /// One connection: one request, answered, and then closed.
    fn handle(&mut self, stream: &mut TcpStream) -> std::io::Result<()> {
        let mut reader = BufReader::new(stream.try_clone()?);
        match read_request(&mut reader).and_then(|request| self.respond(stream, &request)) {
            Ok(log) => {
                eprintln!("{log}");
                Ok(())
            }
            // Every failure is raised before a single byte of the answer is
            // written, so there is always room for the status and the body.
            Err(fail) => write_error(stream, &fail),
        }
    }

    fn respond(&mut self, stream: &mut TcpStream, request: &Request) -> Result<String, Fail> {
        let chat = match (request.method.as_str(), request.path.as_str()) {
            ("POST", "/v1/chat/completions") => true,
            ("POST", "/v1/completions") => false,
            (method, "/v1/chat/completions" | "/v1/completions") => {
                return Err(Fail::new(
                    405,
                    format!("{method} is not allowed on {}, only POST", request.path),
                ));
            }
            (_, path) => {
                return Err(Fail::new(
                    404,
                    format!(
                        "unknown path {path:?}; this server has \
                         POST /v1/chat/completions and POST /v1/completions"
                    ),
                ));
            }
        };
        let body: Value = serde_json::from_slice(&request.body)
            .map_err(|e| Fail::bad(format!("the request body is not JSON: {e}")))?;
        let parsed = Parsed::read(
            &body,
            chat,
            self.tokenizer,
            self.defaults,
            self.runner.context(),
        )?;

        let head = format!(
            "{} {} {}",
            request.method,
            request.path,
            if parsed.stream { "stream" } else { "json" }
        );
        if parsed.stream {
            self.stream_answer(stream, &parsed, chat, &head)
        } else {
            self.whole_answer(stream, &parsed, chat, &head)
        }
    }

    /// Generates, then answers with one object.
    fn whole_answer(
        &mut self,
        stream: &mut TcpStream,
        parsed: &Parsed,
        chat: bool,
        head: &str,
    ) -> Result<String, Fail> {
        let (ids, stop, timings) = self.generate(parsed, |_| true);
        // Decoded in one call rather than per token: a character spread over
        // two tokens cannot be decoded from either alone, and nothing is being
        // streamed here, so there is no reason to try.
        let text = self
            .tokenizer
            .decode(&ids, false)
            .map_err(|e| Fail::new(500, format!("cannot decode what was generated: {e}")))?;

        let answer = Answer::new(chat, &self.defaults.model);
        let body = answer
            .whole(&text, finish_reason(stop), &timings)
            .to_string();
        let log = format!(
            "{head} 200 {}",
            counts(&timings, parsed.sampling.seed, stop)
        );
        if let Err(e) = write_response(stream, 200, "application/json", body.as_bytes()) {
            return Ok(format!("{log}, but the client went away first: {e}"));
        }
        Ok(log)
    }

    /// Generates, sending every token as its own `text/event-stream` event.
    ///
    /// A client that drops mid-stream is a normal end of the request: the
    /// callback stops generation, because nobody is going to read the rest, and
    /// there is nobody left to report it to either.
    fn stream_answer(
        &mut self,
        stream: &mut TcpStream,
        parsed: &Parsed,
        chat: bool,
        head: &str,
    ) -> Result<String, Fail> {
        let answer = Answer::new(chat, &self.defaults.model);
        if let Err(e) = write_header(
            stream,
            200,
            "text/event-stream",
            Some("Cache-Control: no-cache\r\n"),
        ) {
            return Ok(format!(
                "{head} 200, client went away before the first chunk: {e}"
            ));
        }

        let tokenizer = self.tokenizer;
        let mut pending: Vec<u32> = Vec::new();
        let mut events = 0usize;
        let mut gone = None;
        let (_, stop, timings) = self.generate(parsed, |token| {
            pending.push(token);
            // A character spread over two tokens cannot be decoded from either
            // of them alone; the decoder leaves a replacement character where
            // it was. So a token is held back until what is pending decodes
            // into something whole, rather than sending a client bytes it
            // cannot repair.
            let Ok(text) = tokenizer.decode(&pending, false) else {
                return true;
            };
            if text.is_empty() || text.contains(char::REPLACEMENT_CHARACTER) {
                return true;
            }
            pending.clear();
            match answer.chunk(stream, &text, events == 0, None) {
                Ok(()) => {
                    events += 1;
                    true
                }
                Err(e) => {
                    gone = Some(e);
                    false
                }
            }
        });

        // Whatever is left over is a character the model never completed. It
        // goes out as it decodes, replacement character and all, because the
        // alternative is swallowing the end of the answer. Tokens that decode
        // to nothing at all are not an event: there is no delta to send.
        if gone.is_none()
            && !pending.is_empty()
            && let Ok(text) = tokenizer.decode(&pending, false)
            && !text.is_empty()
        {
            match answer.chunk(stream, &text, events == 0, None) {
                Ok(()) => events += 1,
                Err(e) => gone = Some(e),
            }
        }

        let log = format!(
            "{head} 200 {}",
            counts(&timings, parsed.sampling.seed, stop)
        );
        if let Some(e) = gone {
            return Ok(format!(
                "{log}, client went away after {events} chunks: {e}"
            ));
        }
        // The last chunk carries the reason and no content, then the sentinel
        // every OpenAI client stops on.
        if let Err(e) = answer
            .chunk(stream, "", events == 0, Some(finish_reason(stop)))
            .and_then(|()| {
                stream.write_all(b"data: [DONE]\n\n")?;
                stream.flush()
            })
        {
            return Ok(format!("{log}, client went away at the end: {e}"));
        }
        Ok(format!("{log}, {events} chunks"))
    }

    /// One request's generation. The caches are reset first: requests are
    /// independent, and the previous one left its own conversation in them.
    fn generate(
        &mut self,
        parsed: &Parsed,
        on_token: impl FnMut(u32) -> bool,
    ) -> (Vec<u32>, Stop, Timings) {
        self.runner.reset();
        let mut sampler = Sampler::new(self.vocab, parsed.sampling);
        generate::generate(
            self.runner,
            &mut sampler,
            &parsed.prompt,
            parsed.max_tokens,
            on_token,
        )
    }
}

// --------------------------------------------------------------- the request

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

/// Reads one request: the request line, the headers, and exactly as much body
/// as `Content-Length` promises.
///
/// Both lengths are capped, because the other end of a socket is not trusted
/// to stop: an unbounded header line is a way to make the server allocate until
/// it dies.
fn read_request(reader: &mut BufReader<TcpStream>) -> Result<Request, Fail> {
    let mut head = String::new();
    let mut limited = (&mut *reader).take(MAX_HEADER_BYTES);
    loop {
        let start = head.len();
        if limited.read_line(&mut head)? == 0 {
            return Err(Fail::new(
                408,
                "the connection closed before a whole request arrived",
            ));
        }
        if head[start..].trim().is_empty() {
            break;
        }
        if head.len() as u64 >= MAX_HEADER_BYTES {
            return Err(Fail::new(
                431,
                format!("the request line and headers are over {MAX_HEADER_BYTES} bytes"),
            ));
        }
    }

    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split(' ');
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(Fail::bad(format!(
            "cannot read the request line {request_line:?}"
        )));
    };
    if method.is_empty() || target.is_empty() {
        return Err(Fail::bad(format!(
            "cannot read the request line {request_line:?}"
        )));
    }

    let mut length = 0usize;
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                length = value
                    .parse()
                    .map_err(|_| Fail::bad(format!("Content-Length is not a number: {value:?}")))?;
            }
            // Supporting chunked means a second body reader for a shape no
            // OpenAI client sends; refusing it by name beats reading the frame
            // headers as if they were JSON.
            "transfer-encoding" if !value.eq_ignore_ascii_case("identity") => {
                return Err(Fail::bad(format!(
                    "Transfer-Encoding {value:?} is not supported; send a Content-Length"
                )));
            }
            _ => {}
        }
    }
    if length > MAX_BODY_BYTES {
        return Err(Fail::new(
            413,
            format!("the request body is {length} bytes and the limit is {MAX_BODY_BYTES}"),
        ));
    }

    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).map_err(|e| {
        Fail::bad(format!(
            "the request body is shorter than its Content-Length of {length}: {e}"
        ))
    })?;
    Ok(Request {
        method: method.to_string(),
        // A query string is not part of the route, and neither is a fragment,
        // which a client should not send at all but might.
        path: target
            .split(['?', '#'])
            .next()
            .unwrap_or(target)
            .to_string(),
        body,
    })
}

/// What one request asked for, after every field has been checked.
struct Parsed {
    /// The prompt, encoded: the chat template around `messages`, or `prompt`
    /// verbatim.
    prompt: Vec<u32>,
    sampling: Sampling,
    max_tokens: usize,
    stream: bool,
}

impl Parsed {
    /// Reads the honoured fields and leaves the command line's value in place
    /// for every field the request did not send.
    ///
    /// A value that cannot be a setting is refused by name rather than clamped,
    /// the same way the command line refuses its own flags.
    fn read(
        body: &Value,
        chat: bool,
        tokenizer: &Tokenizer,
        defaults: &Defaults,
        context: usize,
    ) -> Result<Self, Fail> {
        if !body.is_object() {
            return Err(Fail::bad("the request body must be a JSON object"));
        }
        let mut sampling = defaults.sampling;
        if let Some(v) = positive(body, "temperature", true)? {
            sampling.temperature = v;
        }
        if let Some(v) = fraction(body, "top_p")? {
            sampling.top_p = v;
        }
        if let Some(v) = fraction(body, "min_p")? {
            sampling.min_p = v;
        }
        if let Some(v) = whole(body, "top_k")? {
            sampling.top_k = v as usize;
        }
        // A penalty of zero would zero every repeated negative logit and divide
        // every positive one by nothing; it is not a setting.
        if let Some(v) = positive(body, "repetition_penalty", false)? {
            sampling.repeat_penalty = v;
        }
        sampling.seed = match whole(body, "seed")? {
            Some(seed) => seed,
            None => defaults.seed.unwrap_or_else(entropy_seed),
        };

        // Capped at the context rather than refused above it. The context is
        // the ceiling on prompt plus generated tokens, so a larger number
        // cannot change what comes back: the run would end at `ContextFull` at
        // exactly the same token. Clients send whatever their own configuration
        // says without knowing this server's context, and refusing a request
        // this server can answer perfectly well would be interop for nothing.
        // It is also what keeps the number a caller can reach bounded.
        let max_tokens = match whole(body, "max_tokens")? {
            Some(0) => return Err(Fail::bad("max_tokens must be at least 1")),
            Some(n) => n.min(context as u64) as usize,
            None => defaults.max_tokens.min(context),
        };

        let prompt = if chat {
            // Qwen clients put the thinking toggle here, which is also where
            // the model's own template reads it.
            let thinking = match body
                .get("chat_template_kwargs")
                .and_then(|k| k.get("enable_thinking"))
            {
                None | Some(Value::Null) => defaults.thinking,
                Some(Value::Bool(b)) => *b,
                Some(v) => {
                    return Err(Fail::bad(format!(
                        "chat_template_kwargs.enable_thinking must be true or false, got {v}"
                    )));
                }
            };
            chat::render(
                &messages(body, defaults)?,
                ChatOptions {
                    add_generation_prompt: true,
                    thinking,
                },
            )
            // A system message that is not first, or a conversation with
            // nothing to answer, is the client's mistake, not the server's.
            .map_err(|e| Fail::bad(e.to_string()))?
        } else {
            match body.get("prompt") {
                Some(Value::String(prompt)) => prompt.clone(),
                Some(v) => {
                    return Err(Fail::bad(format!("prompt must be a string, got {v}")));
                }
                None => return Err(Fail::bad("prompt is required by /v1/completions")),
            }
        };

        let prompt = tokenizer
            .encode(&prompt, true)
            .map_err(|e| Fail::bad(e.to_string()))?;
        if prompt.is_empty() {
            return Err(Fail::bad(
                "the prompt is empty, so there is nothing to generate from",
            ));
        }
        // Refused here rather than generated into. A prompt the context cannot
        // hold would otherwise come back as a 200 with an empty answer, a
        // `finish_reason` of `length` and no counts at all, which a client
        // cannot tell from a model that had nothing to say. Both numbers are in
        // the message, because the client cannot see either of them otherwise.
        if prompt.len() >= context {
            return Err(Fail::bad(format!(
                "the prompt is {} tokens and the context holds {context}, \
                 so there is no room to answer it; send a shorter prompt, or \
                 start the server with a larger --context",
                prompt.len(),
            )));
        }

        Ok(Self {
            prompt,
            sampling,
            max_tokens,
            stream: match body.get("stream") {
                None | Some(Value::Null) => false,
                Some(Value::Bool(b)) => *b,
                Some(v) => return Err(Fail::bad(format!("stream must be true or false, got {v}"))),
            },
        })
    }
}

/// The conversation, with `--system` in front of it when the request sent no
/// system message of its own.
fn messages(body: &Value, defaults: &Defaults) -> Result<Vec<Message>, Fail> {
    let Some(array) = body.get("messages").and_then(Value::as_array) else {
        return Err(Fail::bad(
            "messages is required by /v1/chat/completions and must be an array",
        ));
    };
    let mut out = Vec::with_capacity(array.len() + 1);
    for (i, message) in array.iter().enumerate() {
        let Some(role) = message.get("role").and_then(Value::as_str) else {
            return Err(Fail::bad(format!("messages[{i}] has no role")));
        };
        // Role::from_str is where a role from outside is refused, so a tool or
        // function message is a 400 here instead of a silently dropped turn.
        let role = Role::from_str(role).map_err(|e| Fail::bad(format!("messages[{i}]: {e}")))?;
        let content = match message.get("content") {
            Some(Value::String(content)) => content.clone(),
            // A list of content parts is how an image is sent. Flattening it
            // would drop the image and change the prompt without saying so.
            Some(v) => {
                return Err(Fail::bad(format!(
                    "messages[{i}].content must be a string, got {v}"
                )));
            }
            None => return Err(Fail::bad(format!("messages[{i}] has no content"))),
        };
        out.push(Message::new(role, content));
    }
    if let Some(system) = &defaults.system
        && !out.iter().any(|m| m.role == Role::System)
    {
        out.insert(0, Message::system(system.clone()));
    }
    Ok(out)
}

/// A field that has to be a number if it is there at all.
fn number<'a>(body: &'a Value, field: &str) -> Result<Option<&'a Value>, Fail> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(v) if v.is_number() => Ok(Some(v)),
        Some(v) => Err(Fail::bad(format!("{field} must be a number, got {v}"))),
    }
}

fn whole(body: &Value, field: &str) -> Result<Option<u64>, Fail> {
    let Some(v) = number(body, field)? else {
        return Ok(None);
    };
    v.as_u64().map(Some).ok_or_else(|| {
        Fail::bad(format!(
            "{field} must be a whole number of 0 or more, got {v}"
        ))
    })
}

fn fraction(body: &Value, field: &str) -> Result<Option<f32>, Fail> {
    let Some(v) = finite(body, field)? else {
        return Ok(None);
    };
    if !(0.0..=1.0).contains(&v) {
        return Err(Fail::bad(format!(
            "{field} must be between 0 and 1, got {v}"
        )));
    }
    Ok(Some(v as f32))
}

fn positive(body: &Value, field: &str, zero_allowed: bool) -> Result<Option<f32>, Fail> {
    let Some(v) = finite(body, field)? else {
        return Ok(None);
    };
    if v < 0.0 || (v == 0.0 && !zero_allowed) {
        return Err(Fail::bad(if zero_allowed {
            format!("{field} cannot be negative, got {v}")
        } else {
            format!("{field} must be above 0, got {v}")
        }));
    }
    Ok(Some(v as f32))
}

fn finite(body: &Value, field: &str) -> Result<Option<f64>, Fail> {
    let Some(v) = number(body, field)? else {
        return Ok(None);
    };
    // serde_json only parses a finite number, but an f64 that does not fit an
    // f32 would become an infinity further down.
    match v.as_f64() {
        Some(v) if v.is_finite() && v.abs() <= f32::MAX as f64 => Ok(Some(v)),
        _ => Err(Fail::bad(format!("{field} is out of range: {v}"))),
    }
}

// -------------------------------------------------------------- the response

/// The parts every body of one answer repeats: its id, its timestamp, the model
/// that produced it, and which of the two endpoints shapes it.
struct Answer<'a> {
    id: String,
    created: u64,
    model: &'a str,
    chat: bool,
}

impl<'a> Answer<'a> {
    fn new(chat: bool, model: &'a str) -> Self {
        Self {
            // `entropy_seed` is the standard library's own OS-seeded hasher,
            // which is already here for `--seed`, so an id costs no dependency.
            id: format!(
                "{}-{:016x}",
                if chat { "chatcmpl" } else { "cmpl" },
                entropy_seed()
            ),
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            model,
            chat,
        }
    }

    /// The whole answer as one object.
    fn whole(&self, text: &str, finish: &str, timings: &Timings) -> Value {
        let choice = if self.chat {
            json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            })
        } else {
            json!({"index": 0, "text": text, "logprobs": null, "finish_reason": finish})
        };
        json!({
            "id": self.id,
            "object": if self.chat { "chat.completion" } else { "text_completion" },
            "created": self.created,
            "model": self.model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": timings.prompt_tokens,
                "completion_tokens": timings.generated_tokens,
                "total_tokens": timings.prompt_tokens + timings.generated_tokens,
            },
        })
    }

    /// One `data:` event, flushed on its own.
    fn chunk(
        &self,
        out: &mut impl Write,
        text: &str,
        first: bool,
        finish: Option<&str>,
    ) -> std::io::Result<()> {
        let choice = if self.chat {
            let mut delta = serde_json::Map::new();
            // The role goes on the first chunk of the answer and nowhere else,
            // which is the shape OpenAI's own stream has.
            if first {
                delta.insert("role".to_string(), json!("assistant"));
            }
            if !text.is_empty() {
                delta.insert("content".to_string(), json!(text));
            }
            json!({"index": 0, "delta": Value::Object(delta), "finish_reason": finish})
        } else {
            json!({"index": 0, "text": text, "logprobs": null, "finish_reason": finish})
        };
        let body = json!({
            "id": self.id,
            "object": if self.chat { "chat.completion.chunk" } else { "text_completion" },
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        });
        // One write for the whole event, not one per piece of the format
        // string: a client that goes away must not be left holding half a
        // `data:` line, and an event is a syscall rather than four.
        out.write_all(format!("data: {body}\n\n").as_bytes())?;
        // Per event: a client has to see tokens as they are produced, not one
        // buffer at the end.
        out.flush()
    }
}

/// OpenAI has two reasons that matter here: a stop token is `stop`, and running
/// out of room is `length`, whether the room was `max_tokens` or the context.
fn finish_reason(stop: Stop) -> &'static str {
    match stop {
        Stop::Eos(_) => "stop",
        Stop::MaxTokens | Stop::ContextFull => "length",
        // Only reachable when the client is already gone, so it is never read.
        Stop::Cancelled => "stop",
    }
}

fn counts(timings: &Timings, seed: u64, stop: Stop) -> String {
    format!(
        "prompt {} tokens in {:.2}s, {} generated in {:.2}s = {:.1} tok/s, {stop:?}, seed {seed}",
        timings.prompt_tokens,
        timings.prompt_time.as_secs_f64(),
        timings.generated_tokens,
        timings.generate_time.as_secs_f64(),
        timings.generate_per_second(),
    )
}

/// A request that will not be answered, with the status it gets.
struct Fail {
    status: u16,
    message: String,
}

impl Fail {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }
}

/// Reading the request failed, which is the client's problem either way: a
/// timeout or a half-sent body is not something the server can fix.
impl From<std::io::Error> for Fail {
    fn from(e: std::io::Error) -> Self {
        Self::new(408, format!("could not read the request: {e}"))
    }
}

fn write_error(stream: &mut TcpStream, fail: &Fail) -> std::io::Result<()> {
    let body = json!({
        "error": {
            "message": fail.message,
            "type": if fail.status >= 500 { "server_error" } else { "invalid_request_error" },
            "param": null,
            "code": null,
        }
    })
    .to_string();
    eprintln!("{} {}: {}", fail.status, reason(fail.status), fail.message);
    write_response(stream, fail.status, "application/json", body.as_bytes())
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write_header(
        stream,
        status,
        content_type,
        Some(&format!("Content-Length: {}\r\n", body.len())),
    )?;
    stream.write_all(body)?;
    stream.flush()
}

/// The status line and the headers. A streamed answer has no length to send, so
/// the body's length is a header like any other rather than part of this.
fn write_header(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    extra: Option<&str>,
) -> std::io::Result<()> {
    // Built and then written once: `write!` issues a syscall per piece of the
    // format string, which a client that hangs up can interrupt half-way
    // through the headers.
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\n{}{}Connection: close\r\n\r\n",
        reason(status),
        // Required on a 405, and the two routes only answer POST.
        if status == 405 { "Allow: POST\r\n" } else { "" },
        extra.unwrap_or_default(),
    );
    stream.write_all(head.as_bytes())?;
    stream.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        _ => "Error",
    }
}
