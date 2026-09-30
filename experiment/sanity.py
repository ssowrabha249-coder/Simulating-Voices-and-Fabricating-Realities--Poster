"""Sanity checks for the saved raw outputs. Writes results/sanity_report.txt.

1. Completeness / alignment: every condition has exactly one record per (question_id, participant_id).
2. Parse statistics: valid answers and number of sampling tries.
3. Metric check with synthetic crowds: a random crowd (uniform integers in [0, 2*truth]) must give an
   exact-answer rate near 0 and a large normalised IQR; an all-correct crowd must give rate 1 and IQR 0.
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scoring import question_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))["questions"]
lines = []
say = lambda s="": (print(s), lines.append(s))

say("== 1. Completeness and alignment by ID")
for path in sorted(glob.glob(os.path.join(HERE, "results", "raw", "*.jsonl"))):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    keys = [(r["question_id"], r["participant_id"]) for r in recs]
    cond = recs[0]["condition"]
    per_q = defaultdict(int)
    for q, _ in keys:
        per_q[q] += 1
    ok = len(keys) == len(set(keys)) and set(per_q) == {q["id"] for q in qs} and len(set(per_q.values())) == 1
    truth_ok = all(r["truth"] == {q["id"]: q["truth"] for q in qs}[r["question_id"]] for r in recs)
    say(f"{cond:15s} records={len(recs)} unique={len(set(keys))} per_question={sorted(set(per_q.values()))} "
        f"ids_ok={ok} truth_aligned={truth_ok}")
    valid = sum(r["answer"] is not None for r in recs)
    tries = np.array([r["tries"] for r in recs])
    say(f"{'':15s} valid={valid}/{len(recs)} ({100 * valid / len(recs):.1f}%)  mean tries={tries.mean():.2f}  max tries={tries.max()}")

say()
say("== 2. Synthetic crowds (metric sanity)")
rng = np.random.default_rng(0)
rand_exact, rand_niqr, orc = [], [], []
for q in qs:
    t = q["truth"]
    m = question_metrics(list(rng.integers(0, 2 * t + 1, size=50)), t)
    rand_exact.append(m["exact_rate"]); rand_niqr.append(m["norm_iqr"])
    orc.append(question_metrics([t] * 50, t))
say(f"random crowd : mean exact rate={np.mean(rand_exact):.3f} (expected ~0), median normalised IQR={np.median(rand_niqr):.2f} (expected ~1)")
say(f"all-correct  : exact rate={np.mean([o['exact_rate'] for o in orc]):.3f} (expected 1), max IQR={max(o['iqr'] for o in orc):.1f} (expected 0)")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "sanity_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
