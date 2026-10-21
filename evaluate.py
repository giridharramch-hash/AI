"""
Evaluate the extractive QA system on test_questions.json.

    python evaluate.py                 # full evaluation (downloads the model on first run)
    python evaluate.py --no-model      # retrieval + baseline only (no torch needed)

Metrics (SQuAD-style, computed after lower-casing and removing punctuation/articles):
    EM        exact match with the gold answer
    F1        token-overlap F1 between prediction and gold answer
    Contains  gold answer appears somewhere inside the prediction (lenient)
    Hit@k     a gold answer is inside one of the top-k retrieved passages (retriever only)
"""
import argparse
import json
import re
import string
import time
from collections import Counter, defaultdict

from qa_system import (DEFAULT_MODEL, ExtractiveQA, SentenceBaseline, TfidfRetriever,
                       TransformerReader, load_passages)


# ----------------------------- metrics ------------------------------------- #
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def contains(pred, gold):
    return float(normalize(gold) in normalize(pred))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


class CachedReader:
    """Wraps the reader so the top-k ablation does not re-run the model on the same (question, passage)."""

    def __init__(self, reader):
        self.reader, self.cache = reader, {}

    def read(self, question, context):
        key = (question, context)
        if key not in self.cache:
            self.cache[key] = self.reader.read(question, context)
        return self.cache[key]


def score_system(name, predict, questions, passages):
    """predict(question) -> Answer. Returns summary dict + per-question rows."""
    rows, times = [], []
    for item in questions:
        t0 = time.perf_counter()
        ans = predict(item["question"])
        times.append((time.perf_counter() - t0) * 1000)
        gold_n = normalize(item["answer"])
        if ans.candidates:          # transformer pipeline: check every retrieved passage
            in_retrieved = any(gold_n in normalize(p.text) for p, _, _, _ in ans.candidates)
        else:                       # baseline: the single returned sentence's passage
            in_retrieved = gold_n in normalize(ans.passage.text)
        rows.append({
            "unit": item["unit"], "question": item["question"], "gold": item["answer"],
            "prediction": ans.text, "confidence": round(ans.confidence, 4),
            "EM": exact_match(ans.text, item["answer"]), "F1": f1_score(ans.text, item["answer"]),
            "Contains": contains(ans.text, item["answer"]),
            "source": ans.passage.title or ans.passage.unit,
            "answer_in_retrieved_passages": bool(in_retrieved),
        })
    summary = {
        "EM": round(mean([r["EM"] for r in rows]), 4),
        "F1": round(mean([r["F1"] for r in rows]), 4),
        "Contains": round(mean([r["Contains"] for r in rows]), 4),
        "latency_ms": round(mean(times), 1),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="course_material.txt")
    ap.add_argument("--questions", default="test_questions.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=3, help="main configuration")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    passages = load_passages(args.material)
    questions = json.load(open(args.questions, encoding="utf-8"))
    print(f"{len(passages)} passages, {len(questions)} test questions\n")

    results = {"model": None if args.no_model else args.model, "n_passages": len(passages),
               "n_questions": len(questions), "top_k": args.top_k, "systems": {}}

    # ---- retrieval quality (no model needed) ----
    retriever = TfidfRetriever(passages)
    results["retrieval_hit_at_k"] = {}
    for k in (1, 3, 5):
        hits = [any(normalize(q["answer"]) in normalize(p.text) for p, _ in retriever.search(q["question"], k))
                for q in questions]
        results["retrieval_hit_at_k"][str(k)] = round(mean(hits), 4)
    print("Retriever Hit@k :", results["retrieval_hit_at_k"])

    # ---- baseline: best sentence ----
    baseline = SentenceBaseline(passages)
    summary, base_rows = score_system("baseline", baseline.ask, questions, passages)
    results["systems"]["baseline_sentence_tfidf"] = summary
    print("Baseline        :", summary)

    main_rows = base_rows
    results["main_system"] = "baseline_sentence_tfidf"

    # ---- transformer reader, main config first (un-cached => honest latency), then ablation ----
    if not args.no_model:
        print(f"\nLoading {args.model} ...")
        reader = CachedReader(TransformerReader(args.model))
        for k in [args.top_k] + [k for k in (1, 3, 5) if k != args.top_k]:
            qa = ExtractiveQA(passages, reader, top_k=k)
            summary, rows = score_system(f"top{k}", qa.ask, questions, passages)
            name = f"transformer_top{k}"
            results["systems"][name] = summary
            print(f"Transformer k={k} :", summary, "(latency only meaningful for the first run)")
            if k == args.top_k:
                main_rows = rows
                results["main_system"] = name
        results["systems"][results["main_system"]]["note"] = "latency measured without cache"

    # ---- per-unit breakdown + error analysis for the main system ----
    by_unit = defaultdict(list)
    for r in main_rows:
        by_unit[r["unit"]].append(r)
    results["per_unit"] = {u: {"n": len(rs), "EM": round(mean([r["EM"] for r in rs]), 3),
                               "F1": round(mean([r["F1"] for r in rs]), 3)} for u, rs in by_unit.items()}
    errors = [r for r in main_rows if r["EM"] < 1]
    for e in errors:
        e["error_type"] = ("retrieval miss" if not e["answer_in_retrieved_passages"]
                           else "reader error (wrong or over-long span)")
    results["errors"], results["predictions"] = errors, main_rows

    # ---- print report-ready tables ----
    print("\n| System | EM | F1 | Contains | Latency (ms/question) |\n|---|---|---|---|---|")
    for name, s in results["systems"].items():
        print(f"| {name} | {s['EM']:.3f} | {s['F1']:.3f} | {s['Contains']:.3f} | {s['latency_ms']} |")
    print("\nPer unit (main system):", results["per_unit"])
    print(f"\nErrors for '{results['main_system']}': {len(errors)} of {len(main_rows)}")
    for e in errors[:15]:      # full list is saved in results.json
        print(f"  [{e['error_type']}] Q: {e['question']}\n      gold: {e['gold']!r}  pred: {e['prediction']!r}  conf={e['confidence']}")

    json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
