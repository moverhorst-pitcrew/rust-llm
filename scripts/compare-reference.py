#!/usr/bin/env python3
"""Walk our per-layer activations next to a reference runner's and report where
they start to disagree.

In a 32-block hybrid model a wrong block is invisible in the output: the text
is merely worse. This compares the two implementations node by node, in graph
order, so the first block that drifts is the block to look at.

    rust-llm trace --raw --prompt Hello > ours.txt
    llama-eval-callback -m <model.gguf> -p Hello -n 1 -t 8 -c 512 \\
        --no-repack -fa off -ctk f32 -ctv f32 > ref.log
    scripts/compare-reference.py ref.log ours.txt

Those reference flags matter. With its defaults the reference repacks the
weights, keeps an f16 KV cache and runs flash attention, none of which this
runner does, and the comparison then measures the difference between two
optimisations rather than between two forward passes.

Two signals per node, and they say different things:

* **elem** is the largest relative difference over the first and last three
  elements, which is what the reference prints, with the dump's own rounding
  taken off first. It reads exactly 0 while the two implementations agree in
  every digit the reference printed, so the first nonzero entry is the first
  node where they really differ. This is the one to trust.
* **sum** is the element sum, which the reference prints to six decimals and so
  resolves much further: it sits at 1e-6 or below where **elem** is 0. But a
  residual stream cancels almost completely, so an element-level difference of
  1e-7 lands as 1e-3 of the sum, and a large **sum** on its own means little.

**Read the first block, not the last one.** Through block 0 a correct
implementation keeps **elem** at 0 and **sum** around 1e-6, which is what a
different f32 accumulation order inside the matmul kernels costs. From block 1
on, both columns grow, and on this model they reach 1e-2 by block 3 and stay
there.
That growth is not a bug being found: quantizing activations to int8 is a
discontinuity, so a 1e-6 difference anywhere eventually puts one activation on
the other side of a rounding boundary, and one flipped level moves every output
of that matmul by about 1e-3. From there it compounds. Whether the two
implementations agree is settled by the token-for-token gate in
`crates/rust-llm/tests/gate.rs`, not by this table's bottom rows.

So use it as a bisection tool: a structural error (a swapped head, a missing
norm, a wrong gate) puts **elem** at 1e-1 or worse in one step, out of a
neighbourhood that was two orders of magnitude better. Look for the step, not
for the absolute value.
"""

import re
import sys

NODE = re.compile(r"common_debug_cb_eval:\s+(.*?) = \((\w+)\)\s+(\w+)\(.*\) = \{")
SUM = re.compile(r"^\s+sum = (-?[\d.]+|-?nan|-?inf)\s*$")
VALUES = re.compile(r"^\s*\[\s*(.*?)\s*\],?\s*$")
NUMBER = re.compile(r"-?\d+\.\d+|-?nan|-?inf")


def read_reference(path):
    """[(name, sum, [first three, last three])] in graph order.

    The reference prints a nested bracket dump per tensor and then its sum; it
    omits both for a view it cannot walk, and those nodes are skipped.

    Every printed element is collected, in order, and the first and last three
    taken from that. The dump's shape varies with the tensor's: a 2560-wide row
    prints six values on one line, while a `{1, 32}` tensor prints one value per
    line, so reading a single line is enough for the first case and loses the
    second. The elision in the middle of the dump is what makes this work: the
    values that survive it are exactly the leading and trailing ones.
    """
    out = []
    name = None
    values = []
    for line in open(path, encoding="utf-8", errors="replace"):
        node = NODE.match(line)
        if node:
            name, values = node.group(1), []
            continue
        if name is None:
            continue
        got = SUM.match(line)
        if got:
            out.append((name, float(got.group(1)), values[:3] + values[-3:]))
            name = None
            continue
        row = VALUES.match(line)
        if row:
            values.extend(float(v) for v in NUMBER.findall(row.group(1)))
    return out


def read_ours(path):
    out = []
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            continue
        name, sum_, edges = parts
        out.append((name, float(sum_), [float(v) for v in edges.split(",")]))
    return out


def relative(want, got):
    scale = max(abs(want), abs(got), 1e-6)
    return abs(want - got) / scale


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    reference = read_reference(argv[1])
    ours = read_ours(argv[2])
    # A name can repeat (several nodes are called "Kcur-3"); the first
    # occurrence is the one our trace names.
    by_name = {}
    for name, sum_, values in reference:
        by_name.setdefault(name, (sum_, values))

    print(f"{'node':28s} {'reference':>14s} {'ours':>14s} {'sum':>9s} {'elem':>9s}")
    worst_elem = (0.0, "")
    matched = 0
    for name, sum_, edges in ours:
        found = by_name.get(name)
        if found is None:
            continue
        want_sum, want_edges = found
        matched += 1
        rel_sum = relative(want_sum, sum_)
        rel_elem = 0.0
        if len(want_edges) == len(edges):
            # The reference prints four decimals, so a value near zero has no
            # usable relative precision; scale against the row instead, and
            # ignore the first 1e-4 of every difference, which is the dump's
            # own rounding rather than a disagreement.
            row = max(max(abs(v) for v in want_edges), 1e-3)
            rel_elem = max(
                max(abs(a - b) - 1e-4, 0.0) / row for a, b in zip(want_edges, edges)
            )
        else:
            rel_elem = float("nan")
        if rel_elem > worst_elem[0]:
            worst_elem = (rel_elem, name)
        flag = "  <--" if rel_elem > 1e-3 else ""
        print(
            f"{name:28s} {want_sum:14.6f} {sum_:14.6f} "
            f"{rel_sum:9.2e} {rel_elem:9.2e}{flag}"
        )
    print(f"\n{matched} nodes compared")
    print(f"worst element difference {worst_elem[0]:.2e} at {worst_elem[1]}")
    if not matched:
        print("nothing matched: are the two prompts the same single token?")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
