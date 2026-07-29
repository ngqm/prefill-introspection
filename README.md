# Can LLMs Reliably Self-Report Adversarial Prefills, and How?

[![arXiv](https://img.shields.io/badge/arXiv-2606.23671-b31b1b.svg)](https://arxiv.org/abs/2606.23671)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Content warning:** this repository contains harmful prompts from public safety
benchmarks.

Code for [the paper](https://arxiv.org/abs/2606.23671) by Quang Minh Nguyen,
Uzair Ahmed, and Taegyoon Kim (KAIST), accepted to EMNLP 2026 (Main).

Our paper studies whether language models can verbalize recognition of
adversarial prefills, through two follow-ups: one asking whether the model meant
to write the previous turn, and another asking whether someone altered it. We
work on four research questions:

- **RQ1**: Do models recognize prefilled responses?
- **RQ2**: How do models explain their rejections?
- **RQ3**: Does introspection rely on the refusal direction?
- **RQ4**: Can finetuning improve introspection?

<p align="center">
  <img src="paper/figures/teaser.png" width="62%" alt="A model continues an adversarial prefill with a harmful design, then claims the output as its own.">
</p>

<p align="center"><em>Qwen3-14B continues an adversarial prefill on a SocialHarmBench
prompt, then claims the output as its own.</em></p>

## Setup

```bash
git clone https://github.com/ngqm/prefill-introspection.git
cd prefill-introspection
pip install -r requirements.txt
cp .env.example .env      # fill in OPENROUTER_API_KEY and HF_TOKEN
```

Run every command from the repository root. Outputs go to these folders.

- `rq1_runs/` for generations and labels
- `optimized_prefixes/` for adversarial prefixes
- `activations/` for refusal directions and ablated checkpoints
- `weights/` for LoRA adapters
- `results/` for LaTeX tables
- `paper/figures/` for figures

[`quickstart.ipynb`](quickstart.ipynb) runs one HarmBench prompt through
Llama-3.1-8B with and without a prefill and asks both probes. It needs one GPU
with about 24 GB.

## Data

The four prompt sets ship in `data/`. We removed prompts from JailbreakBench
and StrongREJECT that nearly duplicate prompts in HarmBench or AdvBench.

| Dataset | Source | License |
| --- | --- | --- |
| HarmBench | [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT |
| SocialHarmBench | [psyonp/SocialHarmBench](https://huggingface.co/datasets/psyonp/SocialHarmBench) | Apache-2.0 |
| JailbreakBench | [JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT |
| StrongREJECT | [walledai/StrongREJECT](https://huggingface.co/datasets/walledai/StrongREJECT) | MIT |

The refusal-direction extraction in RQ3 also downloads AdvBench and Alpaca from
the Hugging Face Hub.

## Reproducing the paper

`<model>` is a Hugging Face id such as `Qwen/Qwen3-8B`. `<dataset>` is one of
`harmbench`, `socialharmbench`, `jailbreakbench`, `strongreject`. `<probe>` is
`f1` for the intention probe or `f2` for the tampering probe.

### 1. Generate responses and follow-up replies

```bash
python -m scripts.generate.optimize_adv_prefixes --target-models <model> --datasets <dataset>  # adversarial prefixes
python -m scripts.generate.run_generate --condition static --model <model> --dataset <dataset>  # control and four static prefills
python -m scripts.generate.run_generate --condition adv --model <model> --dataset <dataset>     # adversarial prefill
```

`run_generate` serves the model with vLLM on the local GPU by default.
`--backend openrouter` calls a hosted model, which is how the paper runs
Llama-3.3-70B. `--backend modal` runs the job on [Modal](https://modal.com) and
needs a `huggingface` secret in your Modal workspace.

### 2. Label the replies

```bash
python -m scripts.classify.run_guard                   # Llama Guard 3 1B safety labels
python -m scripts.classify.apply_classifiers_from_gen  # intention and tampering labels for every gen_*.jsonl
```

`apply_classifiers_from_gen` downloads the paper's
[intention](https://huggingface.co/minnq/prefill-introspection-intention-classifier)
and [tampering](https://huggingface.co/minnq/prefill-introspection-tampering-classifier)
classifiers from the Hugging Face Hub. Both are in
[this collection](https://huggingface.co/collections/minnq/can-llms-reliably-self-report-adversarial-prefills-and-how-6ad160de9cbc57a9812fb009).
To train your own classifiers, run the commands below first.
`apply_classifiers_from_gen` then uses the local copies.

```bash
python -m scripts.classify.run_judge                   # DeepSeek V3.2 labels used to stratify the sample
for p in f1 f2; do
  python -m scripts.finetune.build_train_sample --probe $p   # 3,000 replies per probe
  python -m scripts.classify.label_with_gpt41 --probe $p     # GPT-4.1 labels
  python -m scripts.finetune.train_classifier --probe $p     # RoBERTa classifier
done
```

### 3. Reproduce each result

#### RQ1

```bash
python -m src.analysis.bootstrap_gap_se          # recognition gap per model
python -m src.plotting.plot_all_models_baseline  # claim rates per model
```

#### RQ2

```bash
python -m scripts.classify.label_rejection_taxonomy  # rejection reason labels
python -m src.plotting.plot_rejection_reasons        # rejection reason breakdown
```

#### RQ3

Build the refusal-ablated checkpoint and generate with `--ablated`. Then label
the replies as in step 2.

```bash
MODEL_ID=<model> python -m src.pipeline.extract_activations
MODEL_ID=<model> python -m src.pipeline.select_optimal_layer
MODEL_ID=<model> python -m src.pipeline.apply_refusal_ablation
python -m scripts.generate.run_generate --condition static --model <model> --dataset <dataset> --ablated
python -m scripts.generate.run_generate --condition adv --model <model> --dataset <dataset> --ablated
```

The random-direction control uses ten random directions from seeds 42 to 51.
Their checkpoints are `--ablation-kind random`, `random2`, and so on up to
`random10`. Generate with `--ablated --ablation-kind <kind>` for each one. Then
label the replies as in step 2.

```bash
for s in $(seq 42 51); do python -m scripts.ablate.apply_random_ablation --model <model> --seed $s; done
```

Then run the analyses.

```bash
python -m src.plotting.plot_gap_collapse                      # recognition gap under refusal-direction ablation
python -m src.plotting.plot_ablation_asr                      # attack success under refusal-direction ablation
python -m src.analysis.decompose_delta                        # claim rates under refusal-direction ablation
python -m src.analysis.random_direction_gap                   # recognition gap under random-direction ablation
python -m src.analysis.projection_magnitudes --model <model>  # projection magnitudes onto the refusal direction
```

#### RQ4

`<tag>` is `intros_lora_bon` for SFT, `intros_grpo` for GRPO, or `intros_dpo`
for DPO. The held-out split is SocialHarmBench.

```bash
python -m scripts.finetune.build_bon_lora_dataset --model <model>  # best-of-N training data
python -m scripts.finetune.build_lora_dataset --model <model>      # held-out split
python -m scripts.finetune.train_intros_lora --model <model>       # SFT
python -m scripts.finetune.train_intros_grpo --model <model>       # GRPO
python -m scripts.finetune.build_dpo_data --model <model>          # DPO preference pairs
python -m scripts.finetune.train_intros_dpo --model <model>        # DPO
python -m scripts.finetune.eval_intros_lora --model <model> --out-tag <tag> --probe <probe>
python -m scripts.classify.apply_classifiers_from_gen "gen_*__intros_lora_test_*.jsonl"
```

Then run the analyses.

```bash
python -m src.analysis.lora_gap --probe <probe>                      # recognition gap after finetuning
python -m scripts.finetune.asr_lora --model <model> --out-tag <tag>  # attack success after finetuning
```

#### Appendix

```bash
python -m src.plotting.plot_confusion_grids  # probe and refusal agreement
```

The placebo prefills need their own generations.

```bash
python -m scripts.generate.gen_benign_completions --model <model>
python -m scripts.finetune.build_placebo_table
python -m scripts.generate.run_placebo --model <model>
# label the replies as in step 2
python -m src.analysis.analyze_placebo  # claim rates under placebo prefills
```

The human agreement needs your own labels on a sample of replies.

```bash
python -m scripts.classify.sample_for_annotation
python -m scripts.classify.annotate --probe <probe>       # label the sample yourself
python -m scripts.classify.compute_kappa --probe <probe>  # human agreement with the classifiers
```

## Repository layout

```
core/          configuration, model lists, dataset loaders
llm/           vLLM and OpenRouter clients
experiments/   the prefill experiment and the LLM judge
scripts/
  generate/    responses and follow-up replies
  classify/    safety labels, judges, classifiers, annotation
  ablate/      random-direction checkpoints
  finetune/    classifier training and LoRA finetuning
  modal/       the Modal backend
src/
  pipeline/    refusal-direction extraction and ablation
  analysis/    recognition gaps and tables
  plotting/    paper figures
```

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
