//! What one `sample` call allocates, measured over a logit vector the size of
//! this model's vocabulary.
//!
//! Its own test binary, and one test in it, for the same reason
//! `tests/allocations.rs` holds one: the counter below is global and counts
//! every thread, so a second test running next to this one in the harness's
//! thread pool would have its allocations charged to this window. Unlike that
//! file this needs no weights, so it runs in CI.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rust_llm::sample::{Sampler, Sampling};

/// The vocabulary of Qwen3.5-4B, which is the size that makes a per-token
/// allocation expensive in the first place.
const VOCAB: usize = 248_320;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

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

fn allocations(f: impl FnOnce()) -> usize {
    COUNT.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    f();
    COUNTING.store(false, Ordering::Relaxed);
    COUNT.load(Ordering::Relaxed)
}

fn logits(seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..VOCAB)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32 * 12.0 - 6.0
        })
        .collect()
}

/// Every setting that reaches a different stage of the pipeline, because the
/// buffers are sized in `Sampler::new` from the settings and a stage that was
/// skipped proves nothing about the one that was not.
#[test]
fn sampling_allocates_nothing_per_token() {
    let l = logits(3);
    let cases: &[(&str, Sampling)] = &[
        ("greedy", Sampling::GREEDY),
        ("the defaults", Sampling::default()),
        (
            "top-k off, so the candidate set is the whole vocabulary",
            Sampling {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.9,
                min_p: 0.05,
                repeat_penalty: 1.2,
                repeat_last_n: 64,
                seed: 1,
            },
        ),
        (
            "every stage on",
            Sampling {
                temperature: 0.8,
                top_k: 40,
                top_p: 0.9,
                min_p: 0.05,
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                seed: 2,
            },
        ),
    ];

    for &(what, set) in cases {
        let mut sampler = Sampler::new(VOCAB, set);
        // Fill the recent-token window, so the penalty has the most work it
        // will ever have: a full window with a repeat in it. `remember` is
        // what the generation loop calls for the prompt, so this is the state
        // a real run reaches, reached without paying for 64 draws.
        for t in 0..set.repeat_last_n as u32 {
            sampler.remember(t / 2);
        }
        // Then a few real calls, so a lazily initialised anything inside the
        // standard library is not charged to the window below.
        for _ in 0..2 {
            sampler.sample(&l);
        }
        let calls = 8;
        let n = allocations(|| {
            for _ in 0..calls {
                sampler.sample(&l);
            }
        });
        assert_eq!(
            n, 0,
            "{what}: {calls} sample calls over {VOCAB} logits made {n} allocations; \
             every buffer is supposed to come out of Sampler::new"
        );
        println!("{what}: {calls} sample calls, {n} allocations");
    }
}
