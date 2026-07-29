# Can LLMs Reliably Self-Report Adversarial Prefills, and How?

[![arXiv](https://img.shields.io/badge/arXiv-2606.23671-b31b1b.svg)](https://arxiv.org/abs/2606.23671)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Official code for [the paper](https://arxiv.org/abs/2606.23671) by Quang Minh
Nguyen, Uzair Ahmed, and Taegyoon Kim (KAIST). Accepted to EMNLP 2026 (Main).

When an adversarial *prefill* forces a model to begin its reply with harmful
text, can the model recognize this in a follow-up turn? We evaluate ten
open-weight instruction-tuned LLMs (3B to 70B) on four safety benchmarks under
a control and an adversarial-prefill condition, then ask two follow-up
questions: one about *intention* and one about external *tampering*.

- **RQ1**: whether a model recognizes its own prefilled output;
- **RQ2**: a taxonomy of the reasons models give when they reject a prefilled output;
- **RQ3**: refusal-direction ablation (Arditi et al., 2024) as a causal probe, with a random-direction control;
- **RQ4**: whether targeted LoRA finetuning (SFT, GRPO, DPO) widens the gap, and what it costs in MMLU accuracy and attack success rate.

The **recognition gap** is the difference in claim rate between control and
prefilled responses.

<p align="center">
  <img src="paper/figures/teaser.png" width="62%" alt="A model continues an adversarial prefill with a harmful design, then claims the output as its own.">
</p>

<p align="center"><em>Qwen3-14B on a SocialHarmBench prompt: it continues the adversarial prefill,
then claims the output as its own. Across the ten models, the average claim rate
on prefilled responses is 25.3%.</em></p>

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENROUTER_API_KEY and HF_TOKEN
```

## Quickstart

[`quickstart.ipynb`](quickstart.ipynb) walks through the full pipeline on a
single prompt with Llama-3.1-8B: control reply, prefilled reply, both probes.
It needs a GPU and the HarmBench CSV (see Data).

## Data

HarmBench must be downloaded manually; the other datasets are fetched from the
Hugging Face Hub on first use.

| Dataset | Source |
| --- | --- |
| HarmBench | [`harmbench_behaviors_text_test.csv`](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_test.csv) |
| SocialHarmBench | [`psyonp/SocialHarmBench`](https://huggingface.co/datasets/psyonp/SocialHarmBench) |
| JailbreakBench | [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| StrongREJECT | [`walledai/StrongREJECT`](https://huggingface.co/datasets/walledai/StrongREJECT) |
| AdvBench | [`walledai/AdvBench`](https://huggingface.co/datasets/walledai/AdvBench) |

Put the HarmBench CSV at `data/harmbench_behaviors_text_test.csv` (or set
`HARM_BENCH_CSV_PATH`). The JailbreakBench and StrongREJECT loaders also read
this CSV to de-duplicate prompts against HarmBench and AdvBench.

## Backends

The generation scripts take `--backend {local, modal, openrouter}` (default `local`):

- **local** spawns a vLLM server on your GPU.
- **modal** runs the full generation job in one [Modal](https://modal.com) container.
  It needs a Modal account with a `huggingface` secret; fetch outputs with
  `modal volume get introspection-gen /out .`.
- **openrouter** calls hosted models and cannot serve local ablated or finetuned checkpoints.

## Reproducing the paper

Steps 1-3 cover the core pipeline (generate, judge, analyze). Steps 4-6 add the RQ-specific experiments.

1. Generate:
   - `python -m scripts.generate.optimize_adv_prefixes --target-models <m> --datasets <d>` optimizes the adversarial prefixes (local only).
   - `python -m scripts.generate.run_static` generates the control and four static-prefill conditions.
   - `python -m scripts.generate.run_adv` generates the adversarial-prefill condition.
   - The placebo condition (local only) takes three steps:
     `python -m scripts.generate.gen_benign_completions`, then
     `python -m scripts.finetune.build_placebo_table`, then
     `python -m scripts.generate.run_placebo`.

   `run_static` and `run_adv` also accept `--backend modal`. The AdvPrefix optimization and placebo steps are local only.

2. Judge:
   - `python -m scripts.classify.run_guard` adds Llama Guard 3 1B safety labels.
   - Train the F1/F2 reply classifiers (once per probe):
     1. `python -m scripts.classify.run_judge`
     2. `python -m scripts.finetune.build_train_sample --probe {f1,f2}`
     3. `python -m scripts.classify.label_with_gpt41 --probe {f1,f2}`
     4. `python -m scripts.finetune.train_classifier --probe {f1,f2}`
   - `python -m scripts.classify.apply_classifiers_from_gen "gen_*.jsonl"` applies the trained classifiers to all generation files.
3. `python -m src.analysis.bootstrap_gap_se` computes the recognition gap and bootstrap standard errors.
4. `python -m scripts.classify.label_rejection_taxonomy` assigns each rejection to a taxonomy category.
   `python -m src.plotting.plot_rejection_reasons_adv_only` renders the RQ2 figure.
5. RQ3 ablation:
   - Build the ablated checkpoint:
     1. `MODEL_ID=<hf-id> python -m src.pipeline.extract_activations`
     2. `python -m src.pipeline.select_optimal_layer`
     3. `python -m src.pipeline.apply_refusal_ablation`
   - Re-run steps 1-3 with `--ablated --ablation-kind refusal`.
   - For the random-direction control, `python -m scripts.ablate.apply_random_ablation --seed <n>` writes one checkpoint per seed. Re-run steps 1-3 with `--ablation-kind random`.
   - `python -m src.analysis.random_direction_gap --tags <t1,t2,...>` aggregates the gap across seeds and reports the mean and [min, max].
6. RQ4 finetuning:
   - Build training data: `python -m scripts.finetune.build_bon_lora_dataset`.
   - Train (pick one): `python -m scripts.finetune.train_intros_lora --data-tag lora_intros_bon` (SFT),
     `python -m scripts.finetune.train_intros_grpo` (GRPO), or
     `python -m scripts.finetune.build_dpo_data` followed by `python -m scripts.finetune.train_intros_dpo` (DPO).
   - Build the held-out split: `python -m scripts.finetune.build_lora_dataset`.
   - Evaluate: `python -m scripts.finetune.eval_intros_lora`.

## Repository layout

```
core/          shared configuration and dataset loaders
llm/           inference backends and the --backend dispatcher
experiments/   experiment classes (control, static prefill, adversarial prefill)
paper/figures/ figures from the paper

scripts/
  generate/    produce model responses and follow-up replies
  classify/    judge and label replies
  ablate/      build orthogonalized checkpoints and extract directions
  finetune/    LoRA data build, training (SFT/GRPO/DPO), evaluation
  modal/       the --backend modal generation entrypoint

src/
  pipeline/    activation extraction, layer selection, intervention
  analysis/    recognition-gap computation, filters, stats
  plotting/    paper figures and the shared plotting style
```

All scripts run as modules from the repo root, e.g.
`python -m scripts.generate.run_adv --model <hf-id> --dataset harmbench`.
Generated data lands in `rq1_runs/`, `activations/`, `weights/`, and the
classifier directories.

## Responsible use

This repository includes attack tooling (adversarial prefill optimization,
refusal-direction ablation) for reproducing the paper's safety experiments.
Use it only on models you are authorized to evaluate.

## Citation

```bibtex
@misc{nguyen2026can,
  title         = {Can LLMs Reliably Self-Report Adversarial Prefills, and How?},
  author        = {Nguyen, Quang Minh and Ahmed, Uzair and Kim, Taegyoon},
  year          = {2026},
  eprint        = {2606.23671},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {To appear at EMNLP 2026}
}
```

## License

MIT; see [LICENSE](LICENSE).
