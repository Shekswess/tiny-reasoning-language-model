<p align="center">
  <img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5f453496-8180-4cf4-94da-26ebbe1159d4" />
</p>

# Tiny Reasoning Language Model (trlm)

Tiny Reasoning Language Model (trlm) is an open pipeline for teaching a 135M-parameter SmolLM2 based model to handle step-by-step reasoning. The repository captures every stage of the project: 
- sourcing and curating task-specific datasets
- supervised fine-tuning
- preference alignment
- evaluation. 

The code is mainly designed to run on AMD (ROCm) hardware and to publish intermediate artefacts to the Hugging Face Hub, but can be adapted to other setups.
The goal of this project is to demonstrate that even small models can learn to reason with the right training strategy and data.

> [!IMPORTANT]  
> The current model is not intended to be used in everyday applications. It is a research prototype and should be treated as such, mostly because of its limited capabilities, hallucination tendencies, etc.


## Technical Report
For an in-depth explanation of the motivation, training setup, datasets, and findings, see the full [Technical Report]([https://your-report-link-here](https://shekswess.github.io/tiny-reasoning-language-model.html)).

## [Hugging Face Collection](https://huggingface.co/collections/Shekswess/tiny-reasoning-language-model-68d924929c17ad8300544ae4)

<a href="https://huggingface.co/collections/Shekswess/tiny-reasoning-language-model-68d924929c17ad8300544ae4">
  <img width="912" height="786" alt="image" src="https://github.com/user-attachments/assets/35367115-27dd-4262-9f9a-ddc104f4dc31" />
</a>

## Project Stages

| Stage | Objective | Artefact on Hugging Face | Status |
| --- | --- | --- | --- |
| Stage 1 SFT | Teach general intelligence without chain-of-thought | [`Shekswess/trlm-stage-1-sft-final-2`](https://huggingface.co/Shekswess/trlm-stage-1-sft-final-2) | complete (final weights + dataset) |
| Stage 2 SFT | Introduce structured chain-of-thought reasoning with `<think>` tags | [`Shekswess/trlm-stage-2-sft-final-2`](https://huggingface.co/Shekswess/trlm-stage-2-sft-final-2) | complete (final weights + dataset) |
| Stage 3 DPO | Align reasoning style with preference data | [`Shekswess/trlm-stage-3-dpo-final-2`](https://huggingface.co/Shekswess/trlm-stage-3-dpo-final-2) | complete (final weights + dataset) |


## Post-Training Pipeline
<img width="1014" height="563" alt="image" src="https://github.com/user-attachments/assets/195ef389-6aa9-4527-b4f0-bea68c0841ae" />

## Installation and Usage

```bash
# Install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install uv (Windows - PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Check installation
uv --version

# create & sync environment from pyproject.toml
uv sync

# activate virtualenv (if not auto-activated)
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows PowerShell
```

Optional extras live in `pyproject.toml` (`cpu`, `rocm`, `dev`). Install ROCm wheels when targeting AMD accelerators.

Set up credentials before training:

```bash
huggingface-cli login
wandb login 
```

Create a `.env` file if you want to store secrets (loaded automatically by `post_training` scripts).

## Dataset Pipeline

Each stage uses a YAML spec under `data/config/`. The builder streams source datasets, cleans them, logs per-source counts, and can push the shuffled result to the Hub together with an auto-generated dataset card.

### Usage

```bash
uv run data/data_collection.py \
 --config-path data/config/stage_1.yaml \
 --output-dir data/artefacts/stage_1 \
 --upload-to-hub
```

Some specifics that can be controlled via the config:
- Streaming download with optional entry caps per source.
- Column drops / renames (`drop_columns`, `rename_columns`) applied per dataset.
- Automatic `_source_dataset`, `_source_subset`, `_source_split` traceability columns.
- Adaptive strategy: JSON chunking for small corpora, parquet shards for >50k rows.
- Metadata summarising requested vs actual counts and percentage contribution.
- Optional Hugging Face push with README generation (`generate_dataset_card`).

### Stage 1 - Non-Reasoning SFT Blend ([`Shekswess/trlm-sft-stage-1-final-2`](https://huggingface.co/datasets/Shekswess/trlm-sft-stage-1-final-2))

| Source (subset/split) | Samples | Share |
| --- | ---: | ---: |
| HuggingFaceTB/smoltalk2 / smoltalk_smollm3_smol_magpie_ultra_no_think | 33,500 | 57.8% |
| HuggingFaceTB/smoltalk2 / smoltalk_smollm3_smol_summarize_no_think | 7,500 | 12.9% |
| HuggingFaceTB/smoltalk2 / smoltalk_smollm3_smol_rewrite_no_think | 7,500 | 12.9% |
| HuggingFaceTB/smoltalk2 / smoltalk_smollm3_systemchats_30k_no_think | 2,500 | 4.3% |
| HuggingFaceTB/smoltalk2 / smoltalk_smollm3_explore_instruct_rewriting_no_think | 2,500 | 4.3% |
| HuggingFaceTB/smoltalk2 / tulu_3_sft_personas_instruction_following_no_think | 2,500 | 4.3% |
| HuggingFaceTB/smoltalk2 / smoltalk_smollm3_everyday_conversations_no_think | 2,000 | 3.5% |

Focus: everyday conversations, rewrites, summarisation - no chain-of-thought.

### Stage 2 - Reasoning SFT Blend ([`Shekswess/trlm-sft-stage-2-final-2`](https://huggingface.co/datasets/Shekswess/trlm-sft-stage-2-final-2))

| Source (subset/split) | Samples | Share |
| --- | ---: | ---: |
| HuggingFaceTB/smoltalk2 / Llama_Nemotron_Post_Training_Dataset_reasoning_r1 | 40,200 | 51.5% |
| HuggingFaceTB/smoltalk2 / OpenThoughts3_1.2M | 20,000 | 25.6% |
| HuggingFaceTB/smoltalk2 / multi_turn_reasoning_if_think | 10,000 | 12.8% |
| HuggingFaceTB/smoltalk2 / aya_dataset_Qwen3_32B_think | 5,000 | 6.4% |
| HuggingFaceTB/smoltalk2 / smoltalk_everyday_convs_reasoning_Qwen3_32B_think | 2,000 | 2.6% |
| HuggingFaceTB/smoltalk2 / s1k_1.1_think | 800 | 1.0% |

Focus: multi-step reasoning traces with `<think>` delimiters and explicit thought sections. All sources drop `chat_template_kwargs` to keep a uniform schema.

### Stage 3 - Preference Alignment ([`Shekswess/trlm-dpo-stage-3-final-2`](https://huggingface.co/datasets/Shekswess/trlm-dpo-stage-3-final-2))

| Source | Samples | Notes |
| --- | ---: | --- |
| scottgeng00/olmo-3-preference-mix-deltas_reasoning-yolo_scottmix-DECON-chfiltered / train | 50,000 | Drops legacy metadata, renames `dataset` -> `source`. |

Focus: pairwise chosen/rejected reasoning completions for Direct Preference Optimization.

## Training Pipeline

All training scripts accept a `--config-path` pointing to the relevant YAML. Modify hyperparameters by editing the YAML file rather than the Python script.

The whole training process was run on a single AMD MI300x Instance with these specs:
1x 192GB MI300x Virtual Machine
CPU: 8 or 13 cores • RAM: 224GB • Disk: 12288GB NVMe

The following Docker image was used for all stages:

```
docker pull rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.7.1
```

### Stage 1 - Supervised Fine-Tuning (non-reasoning)

```bash
uv run post_training/sft.py \
 --config-path post_training/config/stage_1.yaml
```

- Initial weights: `HuggingFaceTB/SmolLM2-135M-Instruct` (chat tuned).
- Dataset: `Shekswess/trlm-sft-stage-1-final-2` (58k dialogues).
- Chat template: system preamble for `Tiny Reasoning Language Model`, no `<think>` injection.
- Training: 3 epochs, `per_device_train_batch_size=32`, grad accumulation 4, cosine LR (`3e-4` peak), `neftune_noise_alpha=0.01`, BF16/gradient checkpointing.
- Artefacts: checkpoints every 1,500 steps, auto push to `Shekswess/trlm-stage-1-sft-final-2`.

### Stage 2 - Supervised Fine-Tuning (reasoning)

```bash
uv run post_training/sft.py \
 --config-path post_training/config/stage_2.yaml
```

- Initial weights: `Shekswess/trlm-stage-1-sft`.
- Dataset: `Shekswess/trlm-sft-stage-2-final-2` (78k reasoning traces).
- Chat template: forces `<think>...</think>` segments when the system prompt or data indicates reasoning; adds special tokens to the tokenizer before training.
- Training: 1 epoch, same batch/accumulation schedule, stronger `neftune_noise_alpha=0.02`, LR=3e-4, cosine decay.
- Artefacts: saved to `Shekswess/trlm-stage-2-sft-final-2` and pushed on every save.

### Stage 3 - Direct Preference Optimization

```bash
uv run post_training/dpo.py \
 --config-path post_training/config/stage_3.yaml
```

- Initial weights: `Shekswess/trlm-stage-2-sft-final-2`.
- Dataset: `Shekswess/trlm-dpo-stage-3-final-2` (50k chosen/rejected examples).
- Training: 1 epoch, LR=1e-5 with cosine schedule + floor (`min_lr_rate=0.1`), beta=0.1, `apo_zero` objective, grad norm clipped to 0.2.
- Artefacts: output dir `outputs/stage_3_final`, auto-push to `Shekswess/trlm-stage-3-dpo-final-2`.

### Environment & Monitoring

- Mixed precision defaults to BF16;.
- Gradient checkpointing and `dataloader_persistent_workers=true` keep memory in check; adjust worker counts to fit your CPU.
- WANDB logging is enabled by default (`report_to="wandb"`); set `WANDB_PROJECT` etc. in your environment or `.env`.

## Evaluation & Results

- Baseline comparisons with `lm-eval-harness`:
  ```bash
  uv run lm_eval \
  --model_args pretrained=Shekswess/trlm-stage-3-dpo-final-2,trust_remote_code=True,dtype=bfloat16 \
  --tasks gsm8k,bbh,arc_challenge,boolq,piqa,ifeval,mmlu \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path ./results_tiny_reasoning
  ```

| **Benchmark**        | **Tiny Reasoning Language Model (trlm-135M)**  | **SmolLM2-135M-Instruct** | **Improvements** |
| -------------------- | ---------------------------- | ------------------------- | ---------------------------- |
| **ARC Challenge**    | **40.61** (avg)              | 37.3 (avg)                | **+3.31**                    |
| **BBH**              | **36.80** (3-shot)           | 28.2 (3-shot)             | **+8.6**                     |
| **BoolQ**            | **62.17**                    | –                         | N/A                          |
| **GSM8K**            | **2.59** (5-shot)            | 1.4 (5-shot)              | **+1.19**                    |
| **IFEval**           | **35.49** (avg)              | 29.9 (avg)                | **+5.59**                    |
| **MMLU**             | **34.95**                    | 29.3                      | **+5.65**                    |
| **PIQA**             | **64.91**                    | 66.3                      | **–1.39**                    |
| **HellaSwag**        | –                            | 40.9                      | N/A                          |
| **MT-Bench**         | –                            | 19.8                      | N/A                          |


## Potential Improvements & Next Steps

This project is a research prototype, but there are several directions that could strengthen reasoning performance and robustness:

1. **Scale Up Model Size**  
   Train larger backbones in the 250M–300M parameter range, or experiment with architectures optimized for reasoning (e.g., mixture-of-experts, deeper attention layers).

2. **Longer or Multi-Epoch Reasoning SFT**  
   Stage 2 currently runs for a single epoch. Increasing epochs or experimenting with curriculum-style SFT could improve reasoning consistency.

3. **Reinforcement Learning Extensions**  
   Explore GRPO (Generalized Reward Policy Optimization) or other RLHF variants to refine step-by-step reasoning fidelity beyond DPO.

4. **Continued Pretraining with Reasoning Data**  
   Pretrain on synthetic or curated reasoning-heavy corpora before alignment stages, to strengthen inductive biases for reasoning traces.


## Repository Structure

```
.
├── .github
│   ├── workflows
│   │   └── uv_ci.yaml
│   └── dependabot.yml
├── data
│   ├── config
│   │   ├── stage_1.yaml
│   │   ├── stage_2.yaml
│   │   └── stage_3.yaml
│   ├── data_collection.py
├── post_training
│   ├── config
│   │   ├── stage_1.yaml
│   │   ├── stage_2.yaml
│   │   └── stage_3.yaml
│   ├── dpo.py
│   └── sft.py
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── interesting_examples.md
├── pyproject.toml
├── README.md
└── uv.lock
```

## Acknowledgements
- [@HotAisle](https://x.com/HotAisle) for providing the compute resources to train all three stages on a awesome AMD MI300x setup.
- [@mkurman88](https://x.com/mkurman88) for ideas, feedback and code samples.
- [HuggingFaceTB team](https://huggingface.co/HuggingFaceTB) for SmolLM2-135M-Instruct model and the Smoltalk2 dataset collection.
- [@scottgeng00](https://huggingface.co/scottgeng00) for the OLmO-3-Preference-Mix-Deltas dataset.
- [@eliebakouchi](https://x.com/eliebakouch) for help with the tokenization.
