//! The HTTP surface, driven end to end against the tiny committed model.
//!
//! One test, because there is one server: it is started once, every case runs
//! against it in turn, and the child is killed on every path out, a panic
//! included. Needs no weights, so it runs in CI. What it proves is the surface
//! and not the output, like `tiny_model.rs`: the fixture's weights are noise.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::Value;

mod support;

const BIN: &str = env!("CARGO_BIN_EXE_rust-llm");
/// Hard cap on waiting for the server, so one that never listens fails the
/// test instead of hanging the suite.
const WAIT: Duration = Duration::from_secs(60);
/// The command line the server is started with, which is what a request that
/// sends no field of its own inherits.
const DEFAULT_MAX_TOKENS: u64 = 3;
const DEFAULT_SEED: &str = "7";
const DEFAULT_SYSTEM: &str = "Answer in one word.";
const CHAT: &str = "/v1/chat/completions";

#[test]
fn the_http_surface_answers_both_endpoints_and_refuses_bad_input() {
    let tiny = support::write("serve", support::UNREACHED_EOS);
    let server = start(&tiny);
    let addr = server.addr.as_str();

    // ---------------------------------------- a non-streaming chat completion
    let (code, answer) = post(
        addr,
        CHAT,
        r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":6,"temperature":0}"#,
    );
    assert_eq!(code, 200, "{answer}");
    assert_eq!(answer["object"], "chat.completion");
    assert!(
        answer["id"]
            .as_str()
            .expect("an id")
            .starts_with("chatcmpl-"),
        "{answer}"
    );
    assert!(
        answer["created"].as_u64().expect("created") > 1_700_000_000,
        "created is a unix timestamp: {answer}"
    );
    assert_eq!(
        answer["model"], "model.gguf",
        "the model that answered, not the one that was asked for: {answer}"
    );
    let choice = &answer["choices"][0];
    assert_eq!(choice["index"], 0);
    assert_eq!(choice["message"]["role"], "assistant");
    assert!(
        !choice["message"]["content"]
            .as_str()
            .expect("content")
            .is_empty(),
        "nothing was generated: {answer}"
    );
    assert!(
        matches!(choice["finish_reason"].as_str(), Some("stop" | "length")),
        "{answer}"
    );
    // The counts are the real ones: what the prompt was and what came out.
    let usage = &answer["usage"];
    let prompt_tokens = usage["prompt_tokens"].as_u64().expect("prompt_tokens");
    let completion_tokens = usage["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    assert!(prompt_tokens > 0, "{usage}");
    assert!(completion_tokens > 0, "{usage}");
    assert_eq!(
        usage["total_tokens"].as_u64(),
        Some(prompt_tokens + completion_tokens),
        "{usage}"
    );

    // ------------------- the command line's values are the per-request defaults
    let only_messages = r#"{"messages":[{"role":"user","content":"hello"}]}"#;
    let (_, first) = post(addr, CHAT, only_messages);
    let (_, again) = post(addr, CHAT, only_messages);
    assert!(
        first["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens")
            <= DEFAULT_MAX_TOKENS,
        "--max-tokens {DEFAULT_MAX_TOKENS} is the default: {first}"
    );
    assert_eq!(
        first["choices"][0]["message"]["content"], again["choices"][0]["message"]["content"],
        "--seed {DEFAULT_SEED} is the default, so two identical requests draw the same tokens"
    );
    assert_ne!(first["id"], again["id"], "every answer has its own id");

    // ------------------ the system prompt and the thinking toggle over HTTP
    let (_, own_system) = post(
        addr,
        CHAT,
        r#"{"messages":[{"role":"system","content":"x"},{"role":"user","content":"hello"}],
            "max_tokens":1,"temperature":0}"#,
    );
    assert_ne!(
        own_system["usage"]["prompt_tokens"], first["usage"]["prompt_tokens"],
        "a request's own system message replaces --system {DEFAULT_SYSTEM:?}"
    );
    let (_, no_thinking) = post(
        addr,
        CHAT,
        r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0,
            "chat_template_kwargs":{"enable_thinking":false}}"#,
    );
    assert_ne!(
        no_thinking["usage"]["prompt_tokens"], first["usage"]["prompt_tokens"],
        "enable_thinking changes the prompt the model is given"
    );

    // -------------------------------------------- a streamed chat completion
    let (response, event_before_the_end) = stream_post(
        addr,
        CHAT,
        r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":12,"temperature":0,
            "stream":true}"#,
    );
    assert_eq!(status(&response), 200, "{response}");
    let headers = response.to_ascii_lowercase();
    assert!(
        headers.contains("content-type: text/event-stream"),
        "{response}"
    );
    assert!(
        !headers.contains("content-length"),
        "a stream has no length to announce: {response}"
    );
    assert!(
        event_before_the_end,
        "no whole event was in hand before the stream ended: {response}"
    );

    let events = events(&response);
    assert_eq!(
        *events.last().expect("at least one event"),
        "[DONE]",
        "the stream closes with the sentinel every client stops on: {response}"
    );
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|event| serde_json::from_str(event).expect("a chunk is JSON"))
        .collect();
    // Tokens arrive as their own events rather than as one buffer at the end.
    // Fewer events than tokens is not a fault: several of the fixture's tokens
    // decode to nothing, and a token is held back until what is pending
    // decodes into something whole.
    assert!(
        chunks.len() >= 3,
        "twelve tokens were asked for and only {} chunks came back: {response}",
        chunks.len()
    );
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"], "assistant",
        "the role goes on the first chunk: {}",
        chunks[0]
    );
    let id = chunks[0]["id"].as_str().expect("an id").to_string();
    assert!(id.starts_with("chatcmpl-"));
    let (last, content_chunks) = chunks.split_last().expect("chunks");
    for chunk in content_chunks {
        assert_eq!(
            chunk["id"].as_str(),
            Some(id.as_str()),
            "every chunk of one answer carries its id: {chunk}"
        );
        assert!(
            chunk["choices"][0]["finish_reason"].is_null(),
            "a chunk before the end has no reason: {chunk}"
        );
        assert!(
            chunk["choices"][0]["delta"]["content"].is_string(),
            "a content chunk has a delta: {chunk}"
        );
    }
    assert!(
        matches!(
            last["choices"][0]["finish_reason"].as_str(),
            Some("stop" | "length")
        ),
        "the last chunk carries the reason and no content: {last}"
    );
    assert!(last["choices"][0]["delta"]["content"].is_null(), "{last}");

    // The deltas are the whole answer and nothing twice. Fewer events than
    // tokens is how the fixture decodes, so this is what says that holding a
    // token back until it decodes into something whole neither loses text nor
    // repeats it: the same request, not streamed, has to give the same string.
    let streamed: String = content_chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    let (_, whole) = post(
        addr,
        CHAT,
        r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":12,"temperature":0}"#,
    );
    assert_eq!(
        whole["choices"][0]["message"]["content"].as_str(),
        Some(streamed.as_str()),
        "the streamed deltas and the whole answer differ"
    );

    // ------------------- a text completion, whose prompt goes in verbatim
    let (code, answer) = post(
        addr,
        "/v1/completions",
        r#"{"prompt":"hello","max_tokens":4,"temperature":0}"#,
    );
    assert_eq!(code, 200, "{answer}");
    assert_eq!(answer["object"], "text_completion");
    assert!(
        answer["id"].as_str().expect("an id").starts_with("cmpl-"),
        "{answer}"
    );
    assert!(
        !answer["choices"][0]["text"]
            .as_str()
            .expect("text")
            .is_empty(),
        "{answer}"
    );
    assert!(
        answer["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion_tokens")
            > 0,
        "{answer}"
    );

    // ------------------------------- an unknown path and a wrong method
    let (code, error) = post(addr, "/v1/embeddings", "{}");
    assert_eq!(code, 404, "{error}");
    assert_eq!(error["error"]["type"], "invalid_request_error");
    assert!(
        error["error"]["message"]
            .as_str()
            .expect("a message")
            .contains(CHAT),
        "the message names what the server does have: {error}"
    );

    let response = request(addr, "GET", CHAT, "");
    assert_eq!(status(&response), 405, "{response}");
    assert!(
        response.contains("Allow: POST"),
        "a 405 says what is allowed: {response}"
    );

    // ----------------------------- bodies that cannot be honoured, all 400
    for (body, what) in [
        (
            r#"{"messages":[{"role":"user","content":"hi"}"#,
            "a body that is not JSON",
        ),
        ("[1,2,3]", "a body that is not an object"),
        ("{}", "a chat request with no messages"),
        (
            r#"{"messages":[{"role":"tool","content":"hi"}]}"#,
            "a role this runner does not implement",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"late"}]}"#,
            "a system message that is not first",
        ),
        (
            r#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#,
            "content that is not a string",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"temperature":-1}"#,
            "a negative temperature",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"top_p":1.5}"#,
            "a top_p above 1",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"repetition_penalty":0}"#,
            "a repetition penalty of 0",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#,
            "max_tokens of 0",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"seed":-1}"#,
            "a negative seed",
        ),
        (
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":"yes"}"#,
            "a stream flag that is not a boolean",
        ),
    ] {
        let (code, error) = post(addr, CHAT, body);
        assert_eq!(code, 400, "{what} has to be refused: {error}");
        assert_eq!(error["error"]["type"], "invalid_request_error", "{what}");
        assert!(
            !error["error"]["message"]
                .as_str()
                .expect("a message")
                .is_empty(),
            "{what} is refused without saying why"
        );
    }
    let (code, error) = post(addr, "/v1/completions", "{}");
    assert_eq!(code, 400, "a text completion with no prompt: {error}");

    // -------------- a client that drops mid-stream ends that request only
    {
        let mut dropped = connect(addr);
        send(
            &mut dropped,
            addr,
            "POST",
            CHAT,
            r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":512,"temperature":0,
                "stream":true}"#,
        );
        // Read until a whole event has arrived, so the connection is dropped
        // in the middle of the stream and not before it started.
        let mut seen = Vec::new();
        let mut buffer = [0u8; 256];
        while !String::from_utf8_lossy(&seen).contains("\n\n") {
            let read = dropped.read(&mut buffer).expect("read the stream");
            assert!(read > 0, "the stream ended before an event arrived");
            seen.extend_from_slice(&buffer[..read]);
        }
    }
    let (code, answer) = post(
        addr,
        CHAT,
        r#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":2,"temperature":0}"#,
    );
    assert_eq!(
        code, 200,
        "the server answers the next request after a client went away: {answer}"
    );
}

/// The server, with the child killed on every path out of the test, a panic
/// included.
struct Server {
    child: Child,
    addr: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(tiny: &support::Tiny) -> Server {
    let mut child = Command::new(BIN)
        .arg("serve")
        // Port 0, so the test takes whatever is free instead of racing another
        // run for a number, and reads back what it got.
        .args(["--host", "127.0.0.1", "--port", "0"])
        .args(["--max-tokens", "3", "--temperature", "0.8"])
        .args(["--seed", DEFAULT_SEED, "--system", DEFAULT_SYSTEM])
        .arg("--model")
        .arg(&tiny.model)
        .arg("--tokenizer")
        .arg(&tiny.tokenizer)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rust-llm serve");

    let stderr = child.stderr.take().expect("piped stderr");
    let (tx, rx) = mpsc::channel();
    // Drained for the whole run and not only until the address arrives: the
    // server logs a line per request, and a full stderr pipe would block it
    // half-way through the test. Echoed, so a failure shows what it was doing.
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            eprintln!("[server] {line}");
            if let Some(addr) = line.strip_prefix("listening on http://") {
                let _ = tx.send(addr.trim().to_string());
            }
        }
    });

    let addr = rx
        .recv_timeout(WAIT)
        .expect("the server prints the address it is listening on");
    let server = Server { child, addr };
    // Polled rather than slept: that line is printed before the first accept.
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if TcpStream::connect(&server.addr).is_ok() {
            return server;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!(
        "nothing accepted a connection on {} in {WAIT:?}",
        server.addr
    );
}

fn connect(addr: &str) -> TcpStream {
    let stream = TcpStream::connect(addr).expect("connect to the server");
    stream
        .set_read_timeout(Some(WAIT))
        .expect("set a read timeout");
    stream
}

fn send(stream: &mut TcpStream, addr: &str, method: &str, path: &str, body: &str) {
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("send the request");
    stream.flush().expect("flush the request");
}

/// Sends one request on its own connection and returns the whole response.
///
/// One connection per request keeps the client to a few lines: the server
/// answers with `Connection: close`, so reading to the end of the stream is
/// exactly one response.
fn request(addr: &str, method: &str, path: &str, body: &str) -> String {
    let mut stream = connect(addr);
    send(&mut stream, addr, method, path, body);
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("read the response");
    response
}

fn post(addr: &str, path: &str, body: &str) -> (u16, Value) {
    let response = request(addr, "POST", path, body);
    let code = status(&response);
    let value = serde_json::from_str(payload(&response))
        .unwrap_or_else(|e| panic!("the body is not JSON ({e}):\n{response}"));
    (code, value)
}

/// Sends a streaming request and reads it as it arrives, recording whether a
/// whole `data:` event was in hand before the stream ended.
fn stream_post(addr: &str, path: &str, body: &str) -> (String, bool) {
    let mut stream = connect(addr);
    send(&mut stream, addr, "POST", path, body);
    let mut raw = Vec::new();
    let mut buffer = [0u8; 4096];
    let mut event_before_the_end = false;
    loop {
        let read = stream.read(&mut buffer).expect("read the stream");
        if read == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..read]);
        if !event_before_the_end {
            let text = String::from_utf8_lossy(&raw);
            if let Some((_, events)) = text.split_once("\r\n\r\n") {
                event_before_the_end = events.contains("\n\n");
            }
        }
    }
    (
        String::from_utf8_lossy(&raw).into_owned(),
        event_before_the_end,
    )
}

fn status(response: &str) -> u16 {
    response
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status in the response:\n{response}"))
}

/// Everything after the blank line that ends the headers.
fn payload(response: &str) -> &str {
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or_else(|| panic!("no end of headers in the response:\n{response}"))
}

/// The `data:` payloads of an event stream, in order.
fn events(response: &str) -> Vec<&str> {
    payload(response)
        .split("\n\n")
        .map(str::trim)
        .filter(|event| !event.is_empty())
        .map(|event| {
            event
                .strip_prefix("data: ")
                .unwrap_or_else(|| panic!("not an event: {event:?}"))
        })
        .collect()
}
