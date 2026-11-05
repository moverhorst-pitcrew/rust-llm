//! What the generation loop allocates, measured rather than asserted.
//!
//! ```text
//! RUST_LLM_MODEL=<path>/model.gguf \
//!   cargo test -p rust-llm --release --test allocations -- --ignored --nocapture
//! ```
//!
//! It needs the weights, because what is interesting is the real pass over the
//! real shapes: a buffer that is sized from the config is only proven to be
//! sized right by running the model through it.
//!
//! The allocator below counts every call the global allocator sees while the
//! window is open, from any thread, which is what makes it a measurement and
//! not a claim about the code I remembered writing. Counting from every thread
//! is also why this is one test and not two: the counter is global, so a
//! second test running next to it in the harness's thread pool would have its
//! allocations charged to this one's window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use llm_core::Gguf;
use rust_llm::Tokenizer;
use rust_llm::chat::{ChatOptions, Message, render};
use rust_llm::forward::Runner;
use rust_llm::generate::generate;
use rust_llm::model::Model;
use rust_llm::sample::{Sampler, Sampling, argmax};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");

static COUNT: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

/// `System`, plus a counter for the calls made while the window is open.
///
/// Only the three calls that hand out memory are counted; a free is not an
/// allocation and counting it would hide a leak-free loop that allocates.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        tick();
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        tick();
        unsafe { System.realloc(p, l, new) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[inline]
fn tick() {
    if COUNTING.load(Ordering::Relaxed) {
        COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `f` with the counter open and returns how many allocations it made.
///
/// `f` must not print: a `println!` allocates its own buffer and would be
/// counted, correctly, as an allocation.
fn allocations(f: impl FnOnce()) -> usize {
    COUNT.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    f();
    COUNTING.store(false, Ordering::Relaxed);
    COUNT.load(Ordering::Relaxed)
}

/// The model file, from `RUST_LLM_MODEL`.
///
/// No default: a path baked into the repository is a path on one machine. The
/// message is what someone who has just cloned this needs to read.
fn model_path() -> String {
    std::env::var("RUST_LLM_MODEL")
        .unwrap_or_else(|_| panic!("set RUST_LLM_MODEL to the GGUF file (see docs/models.md)"))
}

fn prompt_tokens(tokenizer: &Tokenizer) -> Vec<u32> {
    let prompt = render(
        &[Message::user("Name three colours.")],
        ChatOptions {
            add_generation_prompt: true,
            thinking: false,
        },
    )
    .expect("render");
    tokenizer.encode(&prompt, true).expect("encode")
}

/// Two windows: the hot loop itself, which must allocate nothing at all, and
/// `generate` around it, which allocates a fixed amount set by the call rather
/// than by how many tokens come out of it.
#[test]
#[ignore = "needs the model file"]
fn the_generation_loop_allocates_nothing_per_token() {
    let gguf = Gguf::open(model_path()).expect("open model");
    let model = Model::load(&gguf).expect("load model");
    let tokenizer =
        Tokenizer::from_file(format!("{FIXTURES}/tokenizer.json")).expect("tokenizer.json");
    let prompt = prompt_tokens(&tokenizer);
    let mut runner = Runner::new(&model, 256, 4);

    // The prompt is the warm-up: it runs every block of both kinds, fills the
    // conv and delta-net states and writes the first KV positions, so anything
    // the first pass touches for the first time is already touched.
    let mut next = 0;
    for &t in &prompt {
        next = argmax(runner.forward(t));
    }

    let mut tokens = 0usize;
    let n = allocations(|| {
        for _ in 0..24 {
            next = argmax(runner.forward(next));
            tokens += 1;
        }
    });
    assert_eq!(
        n, 0,
        "{tokens} forward passes made {n} allocations; the pass is supposed to run \
         entirely out of buffers allocated by Runner::new"
    );
    println!("{tokens} forward passes, {n} allocations");

    // Greedy and the full pipeline, because what the sampler costs is a
    // per-call budget too: every one of its buffers comes out of
    // `Sampler::new` and `sample` is supposed to touch nothing else.
    let settings = [
        ("greedy", Sampling::GREEDY),
        (
            "temperature 0.8, top-k 40, top-p 0.9, min-p 0.05, penalty 1.1",
            Sampling {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.05,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                seed: 7,
            },
        ),
    ];

    for (name, set) in settings {
        let mut sampler = Sampler::new(model.vocab(), set);
        // One throwaway call first, so a lazily initialised anything inside
        // the standard library is not charged to the measurements below.
        runner.reset();
        generate(&mut runner, &mut sampler, &prompt, 2, |_| true);

        let mut counts = Vec::new();
        for max_tokens in [4usize, 32] {
            runner.reset();
            let mut produced = 0;
            let n = allocations(|| {
                let (ids, _, _) =
                    generate(&mut runner, &mut sampler, &prompt, max_tokens, |_| true);
                produced = ids.len();
            });
            counts.push((max_tokens, produced, n));
        }

        for (max_tokens, produced, n) in &counts {
            println!("{name}, max_tokens {max_tokens}: {produced} tokens, {n} allocations");
        }
        let (_, _, few) = counts[0];
        let (_, _, many) = counts[1];
        assert_eq!(
            few, many,
            "{name}: generate allocated {few} times for a short run and {many} for a \
             long one, so something in the loop scales with the token count: {counts:?}"
        );
        // The output vector and the copy of the stop-token set, both made once
        // before the loop starts. A third would mean a new allocation in the
        // loop, whatever the settings: the sampler's buffers are allocated by
        // `Sampler::new`, above the window.
        assert!(
            many <= 2,
            "{name}: generate allocated {many} times, expected at most the output \
             vector and the stop-token set"
        );
    }
}
