"""Unit tests for scoring.py. Run: python test_scoring.py"""
from scoring import parse_completion, parse_chat, question_metrics, bootstrap_exact_rate, iqr


def test_parse_completion():
    assert parse_completion("206]") == 206
    assert parse_completion(" 1,064] degrees") == 1064
    assert parse_completion("299 792 458]") == 299792458
    assert parse_completion("206") is None          # no closing bracket -> invalid (Aher rule)
    assert parse_completion("about 200]") is None   # text before number -> invalid
    assert parse_completion("]") is None


def test_parse_chat():
    assert parse_chat("The answer is 206.") == 206
    assert parse_chat("1,064 degrees Celsius") == 1064
    assert parse_chat("I don't know") is None


def test_metrics():
    m = question_metrics([206] * 10, 206)
    assert m["exact_rate"] == 1.0 and m["iqr"] == 0.0 and m["parse_rate"] == 1.0
    m = question_metrics([100, 200, 300, 400, None], 200)
    assert m["n_valid"] == 4 and m["parse_rate"] == 0.8 and m["exact_rate"] == 0.25
    assert iqr([1, 2, 3, 4, 5]) == 2.0


def test_bootstrap():
    p, lo, hi = bootstrap_exact_rate([[1, 1, 1], [2, 2, 2]], [1, 2], n_boot=200)
    assert p == 1.0 and lo == 1.0 and hi == 1.0
    p, lo, hi = bootstrap_exact_rate([[1, 0, 1, 0]], [1], n_boot=500)
    assert p == 0.5 and lo <= 0.5 <= hi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
