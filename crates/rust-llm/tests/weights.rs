//! Checks that the committed tokenizer really is the one these weights were
//! trained with. Needs the GGUF, so it is ignored by default:
//!
//! ```text
//! cargo test -p rust-llm --test weights -- --ignored --nocapture
//! ```

use llm_core::Gguf;
use rust_llm::Tokenizer;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
const DEFAULT_MODEL: &str = "/paperclip/models/qwen3.5-4b-heretic/Qwen3.5-4B-heretic-Q4_K_M.gguf";

fn model() -> Gguf {
    let path = std::env::var("RUST_LLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    Gguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"))
}

#[test]
#[ignore = "needs the model file"]
fn the_committed_tokenizer_matches_the_weights() {
    let g = model();
    let tok =
        Tokenizer::from_file(format!("{FIXTURES}/tokenizer.json")).expect("load tokenizer.json");

    // Every one of the 248077 tokens the tokenizer can produce, against the
    // vocabulary in the weights, plus the 243 padding slots.
    tok.verify_against_weights(&g).expect("vocab agreement");

    let tokens = g.meta_str_array("tokenizer.ggml.tokens").expect("tokens");
    let merges = g.meta_str_array("tokenizer.ggml.merges").expect("merges");
    println!(
        "tokenizer.ggml.model   = {:?}",
        g.meta_str("tokenizer.ggml.model")
    );
    println!(
        "tokenizer.ggml.pre     = {:?}",
        g.meta_str("tokenizer.ggml.pre")
    );
    println!("tokens                 = {}", tokens.len());
    println!("merges                 = {}", merges.len());
    println!("tokenizer.json vocab   = {}", tok.vocab_size());

    assert_eq!(g.meta_str("tokenizer.ggml.model").unwrap(), "gpt2");
    assert_eq!(g.meta_str("tokenizer.ggml.pre").unwrap(), "qwen35");
    assert_eq!(tokens.len(), 248320);
    assert_eq!(merges.len(), 247587);
    assert_eq!(tok.vocab_size(), 248077);

    assert_eq!(g.meta_u32("tokenizer.ggml.eos_token_id").unwrap(), 248046);
    assert_eq!(tok.token_to_id("<|im_end|>"), Some(248046));
    assert_eq!(tok.token_to_id("<|endoftext|>"), Some(248044));
    assert!(
        !g.meta("tokenizer.ggml.add_bos_token")
            .unwrap()
            .as_bool()
            .unwrap()
    );
    assert!(
        g.meta("tokenizer.ggml.normalizer.nfc")
            .unwrap()
            .as_bool()
            .unwrap()
    );

    // The merge list is not reachable through the tokenizers crate, so compare
    // it against the raw JSON. The weights store a merge space-joined, the
    // JSON as a pair.
    let raw: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{FIXTURES}/tokenizer.json")).unwrap(),
    )
    .expect("parse tokenizer.json");
    let json_merges = raw["model"]["merges"].as_array().expect("merges");
    assert_eq!(json_merges.len(), merges.len());
    for (i, (from_weights, from_json)) in merges.iter().zip(json_merges).enumerate() {
        let pair = from_json.as_array().expect("merge pair");
        let joined = format!(
            "{} {}",
            pair[0].as_str().expect("left"),
            pair[1].as_str().expect("right")
        );
        assert_eq!(*from_weights, joined, "merge {i}");
    }
}

#[test]
#[ignore = "needs the model file"]
fn the_committed_tokenizer_is_byte_identical_to_the_models() {
    // Otherwise the committed copy could drift from the file the weights were
    // converted with, and every id check above would be checking the drift
    // against itself.
    let model_dir = std::path::Path::new(DEFAULT_MODEL)
        .parent()
        .expect("model dir")
        .to_path_buf();
    let theirs = std::fs::read(model_dir.join("tokenizer.json")).expect("model tokenizer.json");
    let ours = std::fs::read(format!("{FIXTURES}/tokenizer.json")).expect("committed copy");
    assert_eq!(ours.len(), theirs.len());
    assert!(ours == theirs, "fixtures/tokenizer.json has drifted");
}

#[test]
#[ignore = "needs the model file"]
fn a_different_tokenizer_fails_loudly() {
    let g = model();
    // A three-token BPE. Nothing about it matches these weights, and the point
    // is that we find out by name rather than by generating wrong ids.
    let other = Tokenizer::from_json(
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"BPE","dropout":null,"unk_token":null,
                     "continuing_subword_prefix":null,"end_of_word_suffix":null,
                     "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                     "vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}"#,
    )
    .expect("load the small tokenizer");

    let err = other.verify_against_weights(&g).unwrap_err();
    println!("{err}");
    assert!(err.to_string().contains("token 0"), "{err}");
}
