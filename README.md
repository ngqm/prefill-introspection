# Can LLMs Reliably Self-Report Adversarial Prefills, and How?

[![arXiv](https://img.shields.io/badge/arXiv-2606.23671-b31b1b.svg)](https://arxiv.org/abs/2606.23671)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Official code for [the paper](https://arxiv.org/abs/2606.23671) by Quang Minh
Nguyen, Uzair Ahmed, and Taegyoon Kim (KAIST). Do LLMs recognize, in a follow-up
turn, that a prior response was elicited by an adversarial *prefill* attack? We
evaluate ten open-weight instruction-tuned LLMs (3B to 70B) on four safety
benchmarks under a control and an adversarial-prefill condition, then ask one
follow-up question about *intention* and one about external *tampering*. The
**recognition gap** is the drop in how often a model claims a response as
intended when that response was prefilled. Four research questions:

- **RQ1**: whether a model recognizes its own prefilled output;
- **RQ2**: a taxonomy of the reasons models give when they reject a prefilled output;
- **RQ3**: refusal-direction ablation (Arditi et al., 2024) as a causal probe, with a random-direction control;
- **RQ4**: whether targeted LoRA finetuning (SFT, GRPO, DPO) widens the gap, and what it costs in MMLU accuracy and attack success rate.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENROUTER_API_KEY and HF_TOKEN
```

## Quickstart

[`quickstart.ipynb`](quickstart.ipynb) measures the gap on one prompt with
Llama-3.1-8B: control reply, prefilled reply, both probes. Needs a GPU and the
HarmBench CSV (see Data).

## Data

Download HarmBench manually. The other datasets come from the Hugging Face hub
on first use.

| Dataset | Source |
| --- | --- |
| HarmBench | [`harmbench_behaviors_text_test.csv`](https://github.com/centerforaisafety/HarmBench/blob/main/data/behavior_datasets/harmbench_behaviors_text_test.csv) |
| SocialHarmBench | [`psyonp/SocialHarmBench`](https://huggingface.co/datasets/psyonp/SocialHarmBench) |
| JailbreakBench | [`JailbreakBench/JBB-Behaviors`](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) |
| StrongREJECT | [`walledai/StrongREJECT`](https://huggingface.co/datasets/walledai/StrongREJECT) |
| AdvBench | [`walledai/AdvBench`](https://huggingface.co/datasets/walledai/AdvBench) |

Put the HarmBench CSV at `data/harmbench_behaviors_text_test.csv` (or set
`HARM_BENCH_CSV_PATH`). JailbreakBench and StrongREJECT read it too, to
de-duplicate their prompts against HarmBench and AdvBench.

## Backends

The generation scripts take `--backend {local, modal, openrouter}` (default
`local`). `local` spawns a vLLM server on your GPU. `modal` serves and generates
in one [Modal](https://modal.com) container (needs a Modal account with a
`huggingface` secret; fetch outputs with `modal volume get introspection-gen
/out .`). `openrouter` uses hosted models, so it cannot serve the local ablated or finetuned checkpoints.

## Reproducing the paper

1. Generate:
   - `python -m scripts.generate.optimize_adv_prefixes --target-models <m> --datasets <d>` (local only).
   - `python -m scripts.generate.run_static`: control and the four static prefills.
   - `python -m scripts.generate.run_adv`: the adversarial-prefill condition.
   - Placebo (local): `scripts.generate.gen_benign_completions`, then
     `scripts.finetune.build_placebo_table`, then `scripts.generate.run_placebo`.

   `run_static` and `run_adv` also take `--backend modal` to run on Modal instead of a local GPU; the AdvPrefix and placebo steps are local only.
2. Judge:
   - `python -m scripts.classify.run_guard` first: Llama Guard 3 1B labels.
   - Train the reply classifiers once: `scripts.classify.run_judge`, then
     `scripts.finetune.build_train_sample --probe {f1,f2}`,
     `scripts.classify.label_with_gpt41 --probe {f1,f2}`, and
     `scripts.finetune.train_classifier --probe {f1,f2}`.
   - `python -m scripts.classify.apply_classifiers_from_gen "gen_*.jsonl"`.
3. Analyze: `python -m src.analysis.bootstrap_gap_se`.
4. RQ2 taxonomy: `python -m scripts.classify.label_rejection_taxonomy`, then
   `src.plotting.plot_rejection_reasons_adv_only`.
5. RQ3 ablation: `MODEL_ID=<hf-id> python -m src.pipeline.extract_activations`,
   then `select_optimal_layer`, then `apply_refusal_ablation`; re-run steps 1–3
   with `--ablated --ablation-kind {refusal,random}`. For the random-direction
   control, `scripts.ablate.apply_random_ablation --seed <n>` writes one
   checkpoint per seed and `src.analysis.random_direction_gap --tags
   <t1,t2,...>` reports the mean gap and `[min, max]` over them.
6. RQ4 finetuning: `scripts.finetune.build_bon_lora_dataset`, then
   `scripts.finetune.train_intros_lora --data-tag lora_intros_bon` (SFT),
   `scripts.finetune.train_intros_grpo`, or `scripts.finetune.build_dpo_data`
   followed by `scripts.finetune.train_intros_dpo`. Build the held-out split
   with `scripts.finetune.build_lora_dataset`; evaluate with
   `scripts.finetune.eval_intros_lora`.

## Repository layout

```
core/          config.py (paths, prompts, constants), dataset loaders
llm/           inference backends and the --backend dispatcher
experiments/   experiment classes (control / prefill / static / AdvPrefix)
paper/figures/ the paper's figures (most written by src/plotting)

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

Scripts are Python packages: run them as modules from the repo root, e.g.
`python -m scripts.generate.run_adv --model <hf-id> --dataset harmbench`.
Generated data lands in `rq1_runs/`, `activations/`, `weights/`, and the
classifier directories.

## Responsible use

This repository includes attack tooling (adversarial prefill optimization,
refusal-direction ablation), released so the paper's safety findings can be
reproduced. Use it only on models you are authorized to evaluate.

## Citation

```bibtex
@misc{nguyen2026can,
  title         = {Can LLMs Reliably Self-Report Adversarial Prefills, and How?},
  author        = {Nguyen, Quang Minh and Ahmed, Uzair and Kim, Taegyoon},
  year          = {2026},
  eprint        = {2606.23671},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

## License

MIT; see [LICENSE](LICENSE).
