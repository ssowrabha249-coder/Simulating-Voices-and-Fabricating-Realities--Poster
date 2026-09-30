"""Wisdom-of-Crowds Turing Experiment (after Aher et al., 2023) with a base vs. instruction-tuned model pair.

Conditions
  base_raw      Qwen2.5-0.5B            completion prompt (Aher et al., Fig. 14)
  instruct_raw  Qwen2.5-0.5B-Instruct   identical completion prompt (only the model differs)
  instruct_chat Qwen2.5-0.5B-Instruct   same text sent as a chat message via the model's chat template

Sampling: temperature 1.0, top_p 1.0 (as in Aher et al., p. 7), max 12 new tokens, one sample per participant.
Every raw completion is saved to results/raw/<condition>.jsonl with condition, question id and participant id.

Usage: python run.py [--n 50] [--questions q01,q02] [--out results/raw] [--conditions base_raw,...]
"""
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scoring import parse_completion, parse_chat

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TRIES = 10
MODELS = {"base_raw": "Qwen/Qwen2.5-0.5B", "instruct_raw": "Qwen/Qwen2.5-0.5B-Instruct", "instruct_chat": "Qwen/Qwen2.5-0.5B-Instruct",
          "base15_raw": "Qwen/Qwen2.5-1.5B", "instruct15_raw": "Qwen/Qwen2.5-1.5B-Instruct", "instruct15_chat": "Qwen/Qwen2.5-1.5B-Instruct"}
# 25 common surnames x 2 titles = 50 simulated participants (Aher et al. vary surname and title, p. 2).
SURNAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Nguyen", "Kim", "Patel", "Huang", "Chen",
            "Jackson", "Lee", "Thompson", "White", "Harris"]
PARTICIPANTS = [f"{t} {s}" for s in SURNAMES for t in ("Mr.", "Ms.")]

# Prompt text of Aher et al. (2023), Figure 14, p. 25.
TEMPLATE = ("{name} was asked the following question. They were not allowed to consult any external sources and "
            "were instructed to make their best guess if they were unsure. Their answer was written as an integer "
            "using digits 0-9.\nQuestion (text): [{q}]\n{name}'s answer (integer): [")


def build_inputs(tok, condition, question, names):
    texts = [TEMPLATE.format(name=n, q=question) for n in names]
    if condition.endswith("chat"):
        texts = [tok.apply_chat_template([{"role": "user", "content": t[:-1] + "\nReply with the integer only."}],
                                         tokenize=False, add_generation_prompt=True) for t in texts]
    return texts


def run_condition(condition, questions, names, out_dir, seed_base=1234):
    tok = AutoTokenizer.from_pretrained(MODELS[condition], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODELS[condition], dtype=torch.bfloat16)
    model.eval()
    path = os.path.join(out_dir, f"{condition}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for qi, q in enumerate(questions):
            torch.manual_seed(seed_base + qi)  # fixed seed per (condition, question)
            texts = build_inputs(tok, condition, q["text"], names)
            parse = parse_chat if condition.endswith("chat") else parse_completion
            t0 = time.time()
            # Invalid completions are re-sampled (same prompt, next random draw), at most MAX_TRIES per participant.
            outs, tries = [None] * len(names), [0] * len(names)
            todo = list(range(len(names)))
            while todo and max(tries[i] for i in todo) < MAX_TRIES:
                enc = tok([texts[i] for i in todo], return_tensors="pt", padding=True)
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                         max_new_tokens=12, pad_token_id=tok.pad_token_id or tok.eos_token_id)
                dec = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, o in zip(todo, dec):
                    outs[i], tries[i] = o, tries[i] + 1
                todo = [i for i in todo if parse(outs[i]) is None]
            for pi, (n, o) in enumerate(zip(names, outs)):
                f.write(json.dumps({"condition": condition, "model": MODELS[condition], "question_id": q["id"],
                                    "participant_id": f"p{pi:02d}", "participant": n, "seed": seed_base + qi,
                                    "tries": tries[pi], "completion": o, "answer": parse(o), "truth": q["truth"]}) + "\n")
            print(f"{condition} {q['id']} done in {time.time() - t0:.0f}s; valid "
                  f"{sum(parse(o) is not None for o in outs)}/{len(outs)}; mean tries {sum(tries) / len(tries):.2f}", flush=True)
    del model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--questions", default="")
    ap.add_argument("--conditions", default="base_raw,instruct_raw,instruct_chat")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "raw"))
    a = ap.parse_args()
    qs = json.load(open(os.path.join(HERE, "questions.json"), encoding="utf-8"))["questions"]
    if a.questions:
        qs = [q for q in qs if q["id"] in a.questions.split(",")]
    os.makedirs(a.out, exist_ok=True)
    for c in a.conditions.split(","):
        run_condition(c, qs, PARTICIPANTS[: a.n], a.out)


if __name__ == "__main__":
    main()
