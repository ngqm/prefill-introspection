"""Apply the RoBERTa F1/F2 classifiers to gen_*.jsonl files.

Each reply is truncated to 64 victim-model tokens, as for the judge, and scored
with the classifier of its probe (F1 for *_f1.jsonl, F2 for *_f2.jsonl).
Writes rq1_runs/classified_<name> with every input field plus
`classifier_label` (YES/NO) and `classifier_model`. Existing outputs and
`_shard` partials are skipped.

Usage:
  python -m scripts.classify.apply_classifiers_from_gen ["gen_*.jsonl" ...]
"""
import argparse
import json
from pathlib import Path
from tqdm.auto import tqdm

from core.config import OUT_DIR
from core.utils import read_jsonl
from experiments.judge_evaluator import _clean_and_truncate
from scripts.classify.common import ReplyClassifier, victim_tokenizer

_CLASSIFIERS = {}


def process_gen_file(gen_path: Path):
    """Classify one gen f1/f2 file into OUT_DIR/classified_<name>.

    Return True when a file was written, False when it already existed, and
    None when the file is neither an f1 nor an f2 file.
    """
    fname = gen_path.name
    probe = "f1" if "_f1.jsonl" in fname else "f2" if "_f2.jsonl" in fname else None
    if probe is None:
        return None
    out_path = OUT_DIR / f"classified_{fname}"
    if out_path.exists():
        return False
    if probe not in _CLASSIFIERS:
        _CLASSIFIERS[probe] = ReplyClassifier(probe)
    rows = list(read_jsonl(gen_path))
    tok = victim_tokenizer(fname)
    texts = [_clean_and_truncate((r.get("followup_response") or "").strip(), tok) for r in rows]
    preds = _CLASSIFIERS[probe].predict(texts)
    with out_path.open("w") as f:
        for row, pred in zip(rows, preds):
            f.write(json.dumps({**row, "classifier_label": pred,
                                "classifier_model": f"{probe}_roberta-base"}) + "\n")
    return True


def main():
    """Classify every gen file matched by the glob patterns."""
    ap = argparse.ArgumentParser(description="Apply the F1/F2 reply classifiers to generation files.")
    ap.add_argument("patterns", nargs="*", default=["gen_*.jsonl"],
                    help="Glob patterns under rq1_runs/ (default gen_*.jsonl)")
    args = ap.parse_args()

    files = sorted({p for pat in args.patterns for p in OUT_DIR.glob(pat)})
    files = [p for p in files if "_shard" not in p.name]
    print(f"[*] {len(files)} gen files to process")
    written = skipped = 0
    for p in tqdm(files, desc="classify-from-gen"):
        result = process_gen_file(p)
        if result is True:
            written += 1
        elif result is False:
            skipped += 1
    print(f"[done] wrote {written} classified files; skipped {skipped} existing")


if __name__ == "__main__":
    main()
