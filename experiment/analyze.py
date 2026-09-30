"""Analysis of the saved raw outputs -> results/summary.json, results/tables/*.csv, results/figures/*.pdf.
All numbers on the poster come from these files."""
import csv, glob, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scoring import question_metrics, bootstrap_exact_rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))["questions"]
QIDS = [q["id"] for q in qs]
TRUTH = {q["id"]: q["truth"] for q in qs}
LABEL = {"base_raw": "0.5B base", "instruct_raw": "0.5B instruct", "instruct_chat": "0.5B instruct (chat)",
         "base15_raw": "1.5B base", "instruct15_raw": "1.5B instruct", "instruct15_chat": "1.5B instruct (chat)"}
ORDER = [c for c in LABEL]

data = {}
for path in glob.glob(os.path.join(RES, "raw", "*.jsonl")):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    c = recs[0]["condition"]
    data[c] = {qid: [r["answer"] for r in recs if r["question_id"] == qid] for qid in QIDS}
conds = [c for c in ORDER if c in data]

summary = {"conditions": {}, "tests": {}, "human": {}}
os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
rows = []
for c in conds:
    per_q = {qid: question_metrics(data[c][qid], TRUTH[qid]) for qid in QIDS}
    p, lo, hi = bootstrap_exact_rate([data[c][q] for q in QIDS], [TRUTH[q] for q in QIDS])
    niqr = [per_q[q]["norm_iqr"] for q in QIDS]
    summary["conditions"][c] = {
        "label": LABEL[c], "per_question": per_q,
        "exact_rate": p, "exact_rate_ci95": [lo, hi],
        "n_exact": int(sum(sum(a == TRUTH[q] for a in data[c][q] if a is not None) for q in QIDS)),
        "n_valid": int(sum(per_q[q]["n_valid"] for q in QIDS)),
        "median_norm_iqr": float(np.median(niqr)),
        "n_questions_iqr0": int(sum(per_q[q]["iqr"] == 0 for q in QIDS)),
        "median_abs_rel_error": float(np.median([per_q[q]["median_abs_rel_error"] for q in QIDS])),
    }
    for q in QIDS:
        m = per_q[q]
        rows.append([c, q, TRUTH[q], m["n_valid"], m["median"], m["iqr"], round(m["norm_iqr"], 4), round(m["exact_rate"], 3)])
with open(os.path.join(RES, "tables", "per_question.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["condition", "question", "truth", "n_valid", "median", "iqr", "norm_iqr", "exact_rate"]); w.writerows(rows)
with open(os.path.join(RES, "tables", "overview.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["condition", "exact_rate", "ci_low", "ci_high", "n_exact", "n_valid", "median_norm_iqr", "questions_with_iqr0", "median_abs_rel_error"])
    for c in conds:
        s = summary["conditions"][c]
        w.writerow([c, round(s["exact_rate"], 4), round(s["exact_rate_ci95"][0], 4), round(s["exact_rate_ci95"][1], 4), s["n_exact"], s["n_valid"],
                    round(s["median_norm_iqr"], 4), s["n_questions_iqr0"], round(s["median_abs_rel_error"], 4)])

# Base vs instruction-tuned (same prompt): Fisher exact test on pooled exact answers,
# Wilcoxon signed-rank test on per-question normalised IQR (paired by question, n = 10).
for base, inst in [("base_raw", "instruct_raw"), ("base_raw", "instruct_chat"), ("base15_raw", "instruct15_raw"), ("base15_raw", "instruct15_chat")]:
    if base in data and inst in data:
        sb, si = summary["conditions"][base], summary["conditions"][inst]
        table = [[si["n_exact"], si["n_valid"] - si["n_exact"]], [sb["n_exact"], sb["n_valid"] - sb["n_exact"]]]
        odds, p_f = stats.fisher_exact(table)
        nb = [sb["per_question"][q]["norm_iqr"] for q in QIDS]
        ni = [si["per_question"][q]["norm_iqr"] for q in QIDS]
        try:
            wstat, p_w = stats.wilcoxon(ni, nb)
        except ValueError:
            wstat, p_w = float("nan"), float("nan")
        summary["tests"][f"{inst}_vs_{base}"] = {
            "fisher_exact_rate": {"table_[inst,base]x[exact,not]": table, "odds_ratio": odds, "p": p_f},
            "wilcoxon_norm_iqr": {"statistic": float(wstat), "p": float(p_w), "n_questions_lower_iqr_instruct": int(sum(a < b for a, b in zip(ni, nb)))},
        }

# Human reference (Aher et al., Table 3): first five questions only.
for q in qs[:5]:
    summary["human"][q["id"]] = {"median": q["human_median"], "iqr": q["human_iqr"], "norm_iqr": q["human_iqr"] / q["truth"]}
summary["human_median_norm_iqr_q01_q05"] = float(np.median([summary["human"][q]["norm_iqr"] for q in summary["human"]]))
for c in conds:
    summary["conditions"][c]["median_norm_iqr_q01_q05"] = float(np.median([summary["conditions"][c]["per_question"][q]["norm_iqr"] for q in QIDS[:5]]))

json.dump(summary, open(os.path.join(RES, "summary.json"), "w", encoding="utf-8"), indent=2)

# LaTeX macros with every experiment number used on the poster (generated, never typed by hand).
def pct(x): return f"{100 * x:.1f}\\,\\%"
def pval(p): return "$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"
MAC = {"base_raw": "Base", "instruct_raw": "Inst", "instruct_chat": "Chat"}
tex = ["% generated by analyze.py from results/summary.json -- do not edit"]
for c, m in MAC.items():
    if c not in summary["conditions"]:
        continue
    s = summary["conditions"][c]
    tex += [f"\\newcommand{{\\x{m}Exact}}{{{pct(s['exact_rate'])}}}",
            f"\\newcommand{{\\x{m}ExactCI}}{{{pct(s['exact_rate_ci95'][0])}--{pct(s['exact_rate_ci95'][1])}}}",
            f"\\newcommand{{\\x{m}NExact}}{{{s['n_exact']}}}", f"\\newcommand{{\\x{m}NValid}}{{{s['n_valid']}}}",
            f"\\newcommand{{\\x{m}NIQR}}{{{s['median_norm_iqr']:.2f}}}", f"\\newcommand{{\\x{m}NIQRfive}}{{{s['median_norm_iqr_q01_q05']:.2f}}}",
            f"\\newcommand{{\\x{m}IQRzero}}{{{s['n_questions_iqr0']}}}", f"\\newcommand{{\\x{m}RelErr}}{{{s['median_abs_rel_error']:.2f}}}"]
tex.append(f"\\newcommand{{\\xHumNIQRfive}}{{{summary['human_median_norm_iqr_q01_q05']:.2f}}}")
for key, m in [("instruct_raw_vs_base_raw", "Inst"), ("instruct_chat_vs_base_raw", "Chat")]:
    if key in summary["tests"]:
        t = summary["tests"][key]
        tex += [f"\\newcommand{{\\x{m}FisherP}}{{{pval(t['fisher_exact_rate']['p'])}}}",
                f"\\newcommand{{\\x{m}FisherOR}}{{{t['fisher_exact_rate']['odds_ratio']:.1f}}}",
                f"\\newcommand{{\\x{m}WilcP}}{{{pval(t['wilcoxon_norm_iqr']['p'])}}}",
                f"\\newcommand{{\\x{m}Lower}}{{{t['wilcoxon_norm_iqr']['n_questions_lower_iqr_instruct']}}}"]
open(os.path.join(RES, "numbers.tex"), "w", encoding="utf-8").write("\n".join(tex) + "\n")


# ---------------- figure: (a) exact-answer rate with 95% CI, (b) normalised IQR per question
FLOOR = 1e-3  # IQR = 0 cannot be drawn on a log axis: such bars are drawn to this floor and labelled "0"


def figure(path, w, h, fs, short, fs_min=None, upright=False):
    # fs_min: smallest font size used anywhere in the figure (poster: >= 24 pt at print size, figure included at 100 %)
    fs_min = fs_min or fs * 0.75
    matplotlib.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(w, h), gridspec_kw={"width_ratios": [1, 1.6]})
    cols = {"base": "#9aa5b1", "instruct_raw": "#1f5f8b", "chat": "#0d766e"}
    def col(c): return cols["base"] if "base" in c else (cols["chat"] if c.endswith("chat") else cols["instruct_raw"])
    names = [LABEL[c].replace("0.5B ", "").replace(" (chat)", "\n(chat)") for c in conds]
    vals = [summary["conditions"][c]["exact_rate"] for c in conds]
    err = np.array([[v - summary["conditions"][c]["exact_rate_ci95"][0], summary["conditions"][c]["exact_rate_ci95"][1] - v] for c, v in zip(conds, vals)]).T
    a.bar(range(len(conds)), vals, yerr=err, color=[col(c) for c in conds], capsize=6)
    a.set_xticks(range(len(conds))); a.set_xticklabels(names, fontsize=fs_min)
    # poster (upright=True): no rotated axis labels, the quantity is named in the panel title instead
    if not upright:
        a.set_ylabel("Exact answers (95% CI)", fontsize=fs)
    a.tick_params(labelsize=fs_min)
    a.set_title("(a) Exact answers, 95% CI" if upright else "(a) Share exactly correct", fontsize=fs)
    x = np.arange(len(QIDS))
    wd = 0.8 / (len(conds) + 1)
    for k, c in enumerate(conds):
        v = [summary["conditions"][c]["per_question"][q]["norm_iqr"] for q in QIDS]
        b.bar(x + k * wd, [max(t, FLOOR) for t in v], wd, color=col(c), label=LABEL[c].replace("0.5B ", ""),
              hatch="//" if c.endswith("chat") else None, edgecolor="white")
        for xi, t in zip(x + k * wd, v):
            if t < FLOOR:  # 0 or below the axis floor (Q8: tiny IQR relative to 299,792,458)
                b.text(xi, FLOOR * 1.15, "0" if t == 0 else "<.001", ha="center", va="bottom",
                       fontsize=fs_min, color=col(c))
    b.bar(x[:5] + len(conds) * wd, [summary["human"][q]["norm_iqr"] for q in QIDS[:5]], wd, color="#e07b00", label="humans (Aher et al.)")
    b.set_yscale("log"); b.set_ylim(bottom=FLOOR)
    # plain tick labels: mathtext exponents (10^2) would be drawn at ~70 % of the font size (poster minimum 24 pt)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    b.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1, 10, 100]))
    b.yaxis.set_minor_locator(NullLocator())
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_xticks(x + 0.4 - wd / 2); b.set_xticklabels([f"Q{int(q[1:])}" for q in QIDS], fontsize=fs_min)
    if not upright:
        b.set_ylabel("IQR / true value", fontsize=fs)
    b.tick_params(labelsize=fs_min)
    b.set_title("(b) Spread: IQR / true value per question (log scale)" if upright
                else "(b) Spread of answers per question (log scale)", fontsize=fs)
    b.legend(fontsize=fs_min, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2 if short else 4)
    from matplotlib.ticker import PercentFormatter
    a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for ax in (a, b):
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    print("wrote", path)

os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
figure(os.path.join(RES, "figures", "fig_crowd_poster.pdf"), 21.5, 5.3, 30, False, fs_min=26, upright=True)
print(json.dumps({c: {k: summary["conditions"][c][k] for k in ("exact_rate", "exact_rate_ci95", "median_norm_iqr", "n_questions_iqr0")} for c in conds}, indent=1))
print(json.dumps(summary["tests"], indent=1))
