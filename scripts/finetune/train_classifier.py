"""Train the RoBERTa F1 or F2 reply classifier on GPT-4.1 labels.

Trains on text_truncated (the first 64 victim-model tokens of each reply) with
an 80/20 split stratified by label, and compares held-out accuracy against
GPT-4.1 with that of the run_judge labels used to stratify the sample.

Inputs:  <probe>_classifier/sample_3000_gpt41.jsonl
Outputs: <probe>_classifier/model_<tag>/  (Hugging Face checkpoint, id2label 0=NO, 1=YES)
         <probe>_classifier/eval_<tag>.json

The base model defaults to roberta-base (override with CLF_MODEL); <tag> is its
basename (override with CLF_TAG). The rest of the pipeline loads model_roberta-base.

Usage:
  python -m scripts.finetune.train_classifier --probe {f1,f2}
"""
import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from core.config import ROOT_DIR
from core.utils import read_jsonl

MAX_LEN = 256
SEED = 42


class JudgeDataset(Dataset):
    """Reply texts tokenized once to MAX_LEN, with integer labels (1 = YES)."""

    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length",
                                   max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()} | {"labels": self.labels[i]}


def metrics(pred):
    """Return accuracy, precision, recall, and binary F1 for a Trainer prediction."""
    preds = pred.predictions.argmax(-1)
    p, r, f, _ = precision_recall_fscore_support(pred.label_ids, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(pred.label_ids, preds),
            "precision": p, "recall": r, "f1": f}


def main():
    """Train the classifier, report held-out agreement with GPT-4.1, and save it."""
    parser = argparse.ArgumentParser(description="Train the RoBERTa reply classifier.")
    parser.add_argument("--probe", choices=["f1", "f2"], required=True)
    args_cli = parser.parse_args()

    root = ROOT_DIR / f"{args_cli.probe}_classifier"
    model_name = os.environ.get("CLF_MODEL", "roberta-base")
    out_tag = os.environ.get("CLF_TAG", model_name.split("/")[-1])

    items = [it for it in read_jsonl(root / "sample_3000_gpt41.jsonl") if it["judge_gpt41"] in ("YES", "NO")]
    texts = [it["text_truncated"] for it in items]
    labels = [1 if it["judge_gpt41"] == "YES" else 0 for it in items]
    judge_labels = [1 if it["judge"] == "YES" else 0 for it in items]
    judge_name = (items[0].get("judge_model") if items else None) or "run_judge"

    x_train, x_val, y_train, y_val, judge_train, judge_val, ids_train, ids_val = train_test_split(
        texts, labels, judge_labels, [it["id"] for it in items],
        test_size=0.2, random_state=SEED, stratify=labels,
    )
    print(f"train: {len(x_train)} (YES={sum(y_train)}, NO={len(y_train) - sum(y_train)})")
    print(f"val:   {len(x_val)} (YES={sum(y_val)}, NO={len(y_val) - sum(y_val)})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "NO", 1: "YES"}, label2id={"NO": 0, "YES": 1})
    train_ds = JudgeDataset(tokenizer, x_train, y_train)
    val_ds = JudgeDataset(tokenizer, x_val, y_val)

    out_dir = root / f"model_{out_tag}"
    args = TrainingArguments(
        output_dir=str(out_dir),
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

    val_pred = trainer.predict(val_ds)
    val_preds = val_pred.predictions.argmax(-1)
    cm_clf = confusion_matrix(y_val, val_preds).tolist()
    cm_judge = confusion_matrix(y_val, judge_val).tolist()
    acc_clf = accuracy_score(y_val, val_preds)
    acc_judge = accuracy_score(y_val, judge_val)

    print("\n=== HELD-OUT ACCURACY vs GPT-4.1 ===")
    print(f"  classifier: {acc_clf:.4f}")
    print(f"  {judge_name}: {acc_judge:.4f}")
    print(f"  delta:      {acc_clf - acc_judge:+.4f}")
    for name, cm in (("classifier", cm_clf), (judge_name, cm_judge)):
        print(f"\n=== confusion (rows: gpt-4.1, cols: {name}) ===")
        print("           pred=NO   pred=YES")
        print(f"  gpt=NO   {cm[0][0]:7d}   {cm[0][1]:7d}")
        print(f"  gpt=YES  {cm[1][0]:7d}   {cm[1][1]:7d}")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    eval_summary = {
        "n_train": len(x_train),
        "n_val": len(x_val),
        "classifier_accuracy": acc_clf,
        "judge_accuracy": acc_judge,
        "judge_model": judge_name,
        "classifier_cm": cm_clf,
        "judge_cm": cm_judge,
        "f1": metrics(val_pred)["f1"],
    }
    (root / f"eval_{out_tag}.json").write_text(json.dumps(eval_summary, indent=2))
    print(f"\nsaved model -> {out_dir}")
    print(f"saved eval  -> {root / f'eval_{out_tag}.json'}")


if __name__ == "__main__":
    main()
