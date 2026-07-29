"""Train the RoBERTa F1 or F2 reply classifier on GPT-4.1 labels.

Inputs:  f{1,2}_classifier/sample_3000_gpt41.jsonl
Outputs: f{1,2}_classifier/model_<tag>/  (saved Hugging Face checkpoint)
         f{1,2}_classifier/eval_<tag>.json

Usage: python -m scripts.finetune.train_classifier --probe {f1,f2}
Override the base model with CLF_MODEL (default roberta-base, which is what
produced the shipped f1/f2 classifiers).
"""
import json, os
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--probe", choices=["f1", "f2"], required=True)
args_cli = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2] / f"{args_cli.probe}_classifier"
MODEL_NAME = os.environ.get("CLF_MODEL", "roberta-base")
OUT_TAG   = os.environ.get("CLF_TAG", MODEL_NAME.split("/")[-1])
MAX_LEN = 256
SEED = 42

# Load and split
items = [json.loads(l) for l in (ROOT / "sample_3000_gpt41.jsonl").open()]
items = [it for it in items if it["judge_gpt41"] in ("YES", "NO")]
texts = [it["text_truncated"] for it in items]
labels = [1 if it["judge_gpt41"] == "YES" else 0 for it in items]
judge_labels = [1 if it["judge"] == "YES" else 0 for it in items]

x_train, x_val, y_train, y_val, judge_train, judge_val, ids_train, ids_val = train_test_split(
    texts, labels, judge_labels, [it["id"] for it in items],
    test_size=0.2, random_state=SEED, stratify=labels,
)
print(f"train: {len(x_train)} (YES={sum(y_train)}, NO={len(y_train)-sum(y_train)})")
print(f"val:   {len(x_val)} (YES={sum(y_val)}, NO={len(y_val)-sum(y_val)})")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

class JudgeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()} | {"labels": self.labels[i]}

train_ds = JudgeDataset(x_train, y_train)
val_ds = JudgeDataset(x_val, y_val)

def metrics(pred):
    preds = pred.predictions.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(pred.label_ids, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(pred.label_ids, preds),
            "precision": p, "recall": r, "f1": f}

args = TrainingArguments(
    output_dir=str(ROOT / f"model_{OUT_TAG}"),
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    seed=SEED,
    report_to="none",
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=metrics)
trainer.train()

# Final eval: classifier vs the judge, using GPT-4.1 as ground truth
val_pred = trainer.predict(val_ds)
val_preds = val_pred.predictions.argmax(-1)
cm_clf  = confusion_matrix(y_val, val_preds).tolist()
cm_judge  = confusion_matrix(y_val, judge_val).tolist()
acc_clf = accuracy_score(y_val, val_preds)
acc_judge = accuracy_score(y_val, judge_val)

print(f"\n=== HELD-OUT ACCURACY vs GPT-4.1 ===")
print(f"  classifier:  {acc_clf:.4f}")
print(f"  DeepSeek V3.2 judge: {acc_judge:.4f}")
print(f"  delta:       {acc_clf - acc_judge:+.4f}")

print(f"\n=== confusion (rows: gpt-4.1, cols: classifier) ===")
print(f"           clf=NO   clf=YES")
print(f"  gpt=NO   {cm_clf[0][0]:6d}   {cm_clf[0][1]:6d}")
print(f"  gpt=YES  {cm_clf[1][0]:6d}   {cm_clf[1][1]:6d}")

print(f"\n=== confusion (rows: gpt-4.1, cols: V3.2) ===")
print(f"           v3.2=NO   v3.2=YES")
print(f"  gpt=NO   {cm_judge[0][0]:6d}    {cm_judge[0][1]:6d}")
print(f"  gpt=YES  {cm_judge[1][0]:6d}    {cm_judge[1][1]:6d}")

trainer.save_model(str(ROOT / f"model_{OUT_TAG}"))
tokenizer.save_pretrained(str(ROOT / f"model_{OUT_TAG}"))

eval_summary = {
    "n_train": len(x_train),
    "n_val": len(x_val),
    "classifier_accuracy": acc_clf,
    "judge_accuracy": acc_judge,
    "classifier_cm": cm_clf,
    "judge_cm": cm_judge,
    "f1": metrics(val_pred)["f1"],
}
(ROOT / f"eval_{OUT_TAG}.json").write_text(json.dumps(eval_summary, indent=2))
print(f"\nsaved model -> {ROOT/f'model_{OUT_TAG}'}")
print(f"saved eval  -> {ROOT/f'eval_{OUT_TAG}.json'}")
