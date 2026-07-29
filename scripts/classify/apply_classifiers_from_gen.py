"""Run the RoBERTa F1/F2 classifiers directly on raw gen_*.jsonl files,
writing the canonical `classifier_label` labels.

Outputs: rq1_runs/classified_gen_*.jsonl with rows that preserve every field
from the input gen row plus `classifier_label` and `classifier_model`.
"""
import json, glob, sys
from pathlib import Path
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.classify.run_judge import _victim_hf_id_from_filename, _get_victim_tokenizer
from experiments.judge_evaluator import _clean_and_truncate

ROOT = Path(__file__).resolve().parents[2]
F1_MODEL_DIR = ROOT / "f1_classifier" / "model_roberta-base"
F2_MODEL_DIR = ROOT / "f2_classifier" / "model_roberta-base"
RQ1_DIR = ROOT / "rq1_runs"
MAX_LEN = 256
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] device: {DEVICE}")

def load_clf(model_dir):
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(DEVICE).eval()
    return tok, mdl

f1_tok, f1_mdl = load_clf(F1_MODEL_DIR)
f2_tok, f2_mdl = load_clf(F2_MODEL_DIR)

@torch.inference_mode()
def predict_batch(tok, mdl, texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    logits = mdl(**enc).logits
    return ["YES" if p == 1 else "NO" for p in logits.argmax(-1).cpu().tolist()]

def process_gen_file(gen_path: Path):
    fname = gen_path.name
    is_f1 = "_f1.jsonl" in fname
    is_f2 = "_f2.jsonl" in fname
    if not (is_f1 or is_f2):
        return None
    out_path = RQ1_DIR / f"classified_{fname}"
    if out_path.exists():
        return out_path
    rows = [json.loads(l) for l in gen_path.open()]
    hf = _victim_hf_id_from_filename(fname)
    victim_tok = _get_victim_tokenizer(hf) if hf else None
    texts = [_clean_and_truncate((r.get("followup_response") or "").strip(), victim_tok) for r in rows]
    tok, mdl = (f1_tok, f1_mdl) if is_f1 else (f2_tok, f2_mdl)
    model_id = "f1_roberta-base" if is_f1 else "f2_roberta-base"
    preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        preds.extend(predict_batch(tok, mdl, texts[i:i+BATCH_SIZE]))
    with out_path.open("w") as f:
        for row, pred in zip(rows, preds):
            f.write(json.dumps({**row, "classifier_label": pred, "classifier_model": model_id}) + "\n")
    return out_path

def main():
    # Targets passed via CLI globs; default to all gemma-3 ablated files
    if len(sys.argv) > 1:
        patterns = sys.argv[1:]
    else:
        patterns = [
            "gen_google__gemma-3-4b-it__ablated_*_f[12].jsonl",
            "gen_google__gemma-3-12b-it__ablated_*_f[12].jsonl",
            "gen_adv_google__gemma-3-4b-it__ablated_*_f[12].jsonl",
            "gen_adv_google__gemma-3-12b-it__ablated_*_f[12].jsonl",
        ]
    files = sorted({Path(p) for pat in patterns for p in glob.glob(str(RQ1_DIR / pat))})
    # Exclude shard partials (they get merged separately)
    files = [p for p in files if "_shard" not in p.name]
    print(f"[*] {len(files)} gen files to process")
    written = skipped = 0
    for p in tqdm(files, desc="classify-from-gen"):
        out = process_gen_file(p)
        if out is None: continue
        if out.exists() and out.stat().st_mtime > p.stat().st_mtime:
            written += 1
        else:
            skipped += 1
    print(f"[done] wrote/refreshed {written} classified files; skipped {skipped}")

if __name__ == "__main__":
    main()
