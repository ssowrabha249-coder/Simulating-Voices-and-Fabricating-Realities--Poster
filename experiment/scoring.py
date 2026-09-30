"""Parsing and metrics for the wisdom-of-crowds replication.

Parsing rule for completion prompts follows Aher et al. (2023, Fig. 14 caption, p. 25): a valid completion
is an integer (commas and spaces ignored) followed by a closing bracket "]". For the chat condition there is
no bracket, so the first integer in the reply is used.
"""
import re
import numpy as np

_BRACKET = re.compile(r"^\s*([0-9][0-9, ]*)\]")
_FIRST_INT = re.compile(r"[0-9][0-9,]*")


def parse_completion(text):
    """Return the integer of a valid bracketed completion, else None."""
    m = _BRACKET.match(text)
    if not m:
        return None
    digits = m.group(1).replace(",", "").replace(" ", "")
    return int(digits) if digits else None


def parse_chat(text):
    """Return the first integer in a free-text reply (commas ignored), else None."""
    m = _FIRST_INT.search(text)
    if not m:
        return None
    digits = m.group(0).replace(",", "")
    return int(digits) if digits else None


def iqr(values):
    v = np.asarray(values, dtype=float)
    return float(np.percentile(v, 75) - np.percentile(v, 25))


def question_metrics(answers, truth):
    """Metrics for one question: answers = list of parsed integers (None = invalid)."""
    valid = [a for a in answers if a is not None]
    out = {"n": len(answers), "n_valid": len(valid), "parse_rate": len(valid) / len(answers) if answers else 0.0}
    if valid:
        v = np.asarray(valid, dtype=float)
        out.update({
            "median": float(np.median(v)),
            "iqr": iqr(v),
            "norm_iqr": iqr(v) / truth,
            "exact_rate": float(np.mean(v == truth)),
            "median_abs_rel_error": float(np.median(np.abs(v - truth) / truth)),
        })
    return out


def bootstrap_exact_rate(per_question_answers, truths, n_boot=10000, seed=0):
    """95% CI of the pooled exact-answer rate, resampling answers within each question (stratified)."""
    rng = np.random.default_rng(seed)
    arrays = [np.asarray([a == t for a in ans if a is not None], dtype=float)
              for ans, t in zip(per_question_answers, truths)]
    arrays = [a for a in arrays if len(a)]
    point = float(np.concatenate(arrays).mean())
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = np.concatenate([rng.choice(a, size=len(a), replace=True) for a in arrays]).mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)
