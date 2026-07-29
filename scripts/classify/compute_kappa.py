"""Report Cohen's kappa between the human labels and the two pipeline judges.

Reads rq1_runs/human_labels_<probe>.jsonl (SKIP rows dropped) and the
annotation sample it was drawn from. RoBERTa labels come from the
classified_gen_* file holding each reply. GPT-4.1 labels come from
<probe>_classifier/sample_3000_gpt41.jsonl, so that comparison covers only the
annotated replies that are also in the classifier training sample. Replies are
matched on (model, dataset, prompt_id, condition, reply text).

Usage:
  python -m scripts.classify.compute_kappa --probe {f1,f2}
"""
import argparse

from sklearn.metrics import accuracy_score, cohen_kappa_score

from core.config import OUT_DIR, ROOT_DIR
from core.utils import read_jsonl


def reply_key(model, dataset, row):
    """Return the key that identifies one follow-up reply across files."""
    return (model, dataset, row.get("prompt_id"), row.get("condition"),
            (row.get("followup_response") or "").strip())


def roberta_index(sample, probe):
    """Index classifier_label by reply key over the classified files the sample draws from."""
    idx = {}
    sources = {(r["model_name"], r["dataset"], r["condition"] == "adv_prefill") for r in sample}
    for model, dataset, adv in sources:
        path = OUT_DIR / f"classified_gen{'_adv' if adv else ''}_{model}_{dataset}_{probe}.jsonl"
        for r in read_jsonl(path):
            idx[reply_key(model, dataset, r)] = r.get("classifier_label")
    return idx


def gpt41_index(probe):
    """Index judge_gpt41 by reply key over the classifier training sample."""
    path = ROOT_DIR / f"{probe}_classifier" / "sample_3000_gpt41.jsonl"
    return {reply_key(r["model"], r["dataset"], r): r.get("judge_gpt41") for r in read_jsonl(path)}


def report(name, pairs):
    """Print n, accuracy, and kappa for (human, judge) label pairs."""
    if not pairs:
        print(f"  {name}: no overlapping labels")
        return
    human, judge = zip(*pairs)
    print(f"  {name}: n={len(pairs)}  acc={accuracy_score(human, judge):.3f}  "
          f"kappa={cohen_kappa_score(human, judge):.3f}")


def main():
    """Report kappa and accuracy of RoBERTa and GPT-4.1 against the human labels."""
    ap = argparse.ArgumentParser(description="Cohen's kappa between human labels and the pipeline judges.")
    ap.add_argument("--probe", choices=["f1", "f2"], required=True)
    args = ap.parse_args()
    probe = args.probe

    labels = {r["reply_id"]: r["label"] for r in read_jsonl(OUT_DIR / f"human_labels_{probe}.jsonl")
              if r.get("label") in ("YES", "NO")}
    if not labels:
        print(f"No human labels for {args.probe} yet.")
        return
    sample = [r for r in read_jsonl(OUT_DIR / f"annotation_sample_{probe}.jsonl") if r["reply_id"] in labels]

    judges = {"RoBERTa": roberta_index(sample, probe), "GPT-4.1": gpt41_index(probe)}
    print(f"=== {args.probe}  human n={len(labels)} ===")
    for name, idx in judges.items():
        pairs = []
        for r in sample:
            label = idx.get(reply_key(r["model_name"], r["dataset"], r))
            if label in ("YES", "NO"):
                pairs.append((labels[r["reply_id"]], label))
        report(f"{name} vs. human", pairs)


if __name__ == "__main__":
    main()
