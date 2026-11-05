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
use rust_llm::generate::{generate, generate_drafting};
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
    // The largest batch this test uses, so `Runner::new` sizes the scratch for
    // all of them and the batched windows below are measured against buffers
    // that already exist.
    const BATCH: usize = 8;
    /// Drafts per pass in the speculated window at the end.
    const DRAFT: usize = 4;
    let mut runner = Runner::with_batch(&model, 256, 4, BATCH);

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

    // The same for the batched prompt path, which has its own scratch: `t`
    // rows of every buffer and `t` rows of activation blocks, all sized by
    // `Runner::new`. Several batch sizes, because what would allocate is a
    // buffer sized from the batch rather than from the maximum, and a batch
    // smaller than the maximum would not catch that.
    let mut batched = Vec::new();
    for batch in [1usize, 3, BATCH] {
        runner.reset();
        let mut read = 0;
        let n = allocations(|| {
            while read < prompt.len() {
                let take = (prompt.len() - read).min(batch);
                runner.forward_batch(&prompt[read..read + take]);
                read += take;
            }
        });
        batched.push((batch, read, n));
    }
    for (batch, read, n) in &batched {
        println!("batch {batch}: {read} prompt tokens, {n} allocations");
    }
    assert!(
        batched.iter().all(|&(_, _, n)| n == 0),
        "a batched prompt allocated: {batched:?}"
    );

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

    // And the speculated path, which adds three buffers of its own and a
    // drafter that scans the whole history every pass. Every one of those is
    // something the loop could have been written to allocate per token, so this
    // is the window that says it does not.
    //
    // The drafts are injected and every one of them is accepted, because what
    // has to be covered is a pass that runs a verification batch and then reads
    // several rows out of it. A prompt-lookup draft on this prompt would mostly
    // be no draft at all, and that path is the plain one already measured above.
    let mut runner = Runner::with_speculation(&model, 256, 4, BATCH, DRAFT);
    let mut sampler = Sampler::greedy(model.vocab());
    // The tokens a sequential run produces, which the drafter then offers back.
    let (truth, _, _) = generate(&mut runner, &mut sampler, &prompt, 32, |_| true);
    let mut known: Vec<u32> = prompt.clone();
    known.extend_from_slice(&truth);
    known.extend(std::iter::repeat_n(0, DRAFT + 1));

    let mut speculated = Vec::new();
    for max_tokens in [4usize, 32] {
        let mut sampler = Sampler::greedy(model.vocab());
        // The throwaway call, for the same reason as above.
        runner.reset();
        generate(&mut runner, &mut sampler, &prompt, 2, |_| true);
        runner.reset();
        let mut produced = 0;
        let mut rows = 0;
        let n = allocations(|| {
            let (ids, _, timings) = generate_drafting(
                &mut runner,
                &mut sampler,
                &prompt,
                max_tokens,
                |seen, out| {
                    let take = out.len().min(known.len() - seen.len());
                    out[..take].copy_from_slice(&known[seen.len()..][..take]);
                    take
                },
                |_| true,
            );
            produced = ids.len();
            rows = timings.speculation.passes() + timings.speculation.saved();
        });
        speculated.push((max_tokens, produced, rows, n));
    }
    for (max_tokens, produced, rows, n) in &speculated {
        println!(
            "speculated by {DRAFT}, max_tokens {max_tokens}: {produced} tokens over \
             {rows} verification rows, {n} allocations"
        );
    }
    assert!(
        speculated.iter().all(|&(_, _, rows, _)| rows > 0),
        "no verification batch ran, so the speculated loop was not measured: \
         {speculated:?}"
    );
    let (_, _, _, few) = speculated[0];
    let (_, _, _, many) = speculated[1];
    assert_eq!(
        few, many,
        "the speculated loop allocated {few} times for a short run and {many} for a \
         long one, so something in it scales with the token count: {speculated:?}"
    );
    // The output vector, the stop-token set, the token history, the draft
    // buffer and the row buffer: five, each made once before the loop starts.
    assert!(
        many <= 5,
        "the speculated loop allocated {many} times, expected at most the output \
         vector, the stop-token set and the three buffers speculation adds"
    );
}
