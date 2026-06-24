# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConflictScope is the evaluation pipeline from [Generative Value Conflicts Reveal LLM Priorities](https://www.arxiv.org/abs/2509.25369). It generates value-conflict scenarios, evaluates how models prioritize competing values (via multiple-choice or simulated-user conversations), and produces model rankings.

This is a clean clone of `github.com/andyjliu/conflictscope`, set up under the `agent-values` project to run tests on the ConflictScope data.

## Environment

Use the dedicated conda env **`conflictscope-av`** (Python 3.12). It is separate from the older `conflictscope` env so this project stays isolated.

```bash
conda activate conflictscope-av
```

Dependencies were originally installed from `requirements.txt` (vllm 0.9.1, torch 2.7.0, openai, anthropic, choix, transformers, sentence-transformers, statsmodels, wandb). **The env has since been upgraded** to vllm 0.22.0+cu129 / torch 2.11.0+cu128 / transformers 5.10.1 to support recent model architectures — see ["Upgraded env for recent model architectures"](#upgraded-env-for-recent-model-architectures-qwen35-gemma-4-olmo-3) below.

API keys must be set as conda env vars (not yet configured — set your own):
```bash
conda env config vars set OPENAI_API_KEY=<>
conda env config vars set ANTHROPIC_API_KEY=<>
# optional, for open-weight inference via vLLM:
conda env config vars set HF_HOME=/data/tir/projects/tir3/users/andyliu HF_API_KEY=<>
```
Open-weight models run through vLLM; add models/backends by editing `src/model_wrappers.py`.

### Upgraded env for recent model architectures (Qwen3.5, gemma-4, Olmo-3)

The stock env (vllm 0.9.1 / torch 2.7.0 / transformers 4.52.4) **cannot load** newer
architectures — transformers errors with `model type 'qwen3_5'/'gemma4'/'olmo3' not
recognized`. We upgraded `conflictscope-av` in place. The order and CUDA build matter
because Babel's GPU driver is **CUDA 12.9**, while the default PyPI wheels for the new
vLLM/torch are **CUDA 13** (which fail at runtime with `driver too old` /
`libcudart.so.13: cannot open shared object file`).

Working combo: **vllm 0.22.0+cu129 · torch 2.11.0+cu128 · transformers 5.10.1**. Steps:

```bash
conda activate conflictscope-av

# 1. Upgrade vLLM (pulls torch 2.11 + transformers 4.57; default build is CUDA 13).
pip install -U vllm

# 2. Bump transformers so the new archs register (4.57 has olmo3 but NOT qwen3_5/gemma4).
pip install -U transformers                       # -> 5.10.1

# 3. Replace the CUDA-13 builds with CUDA-12.x ones that match the 12.9 driver.
#    torch -> cu128:
uv pip install -U "vllm==0.22.0" --torch-backend=cu128 --python "$(which python)"
#    vLLM's own compiled ext (vllm._C) is still CUDA 13 after the above; --torch-backend
#    only changes torch. There is no cu128 wheel for 0.22.0, but there is a cu129 one,
#    and cu129 matches the driver. Install it directly from the GitHub release:
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.22.0/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --python "$(which python)"

# Verify: should print versions and "vllm._C import OK" with no libcudart error.
python -c "import torch,transformers,vllm,vllm._C; print(torch.__version__, transformers.__version__, vllm.__version__)"
```

Leftover harmless warnings: `xformers 0.0.30` pins torch 2.7 and a stray
`nvidia-cuda-runtime` (CUDA 13) remains installed — neither is on the MCQ load path.

### Running open-weight evals on SLURM (Babel)

- **Partition `general`, not `array`.** The `array` pool includes RTX PRO 6000 Blackwell
  (sm_120) nodes that the torch builds above don't support (`no kernel image is
  available`). Pin a known-good GPU, e.g. `--gres=gpu:A6000:1`.
- **Export `VLLM_WORKER_MULTIPROC_METHOD=spawn`.** vLLM 0.22's v1 engine otherwise dies
  with `Cannot re-initialize CUDA in forked subprocess`.
- vLLM writes progress bars to **stderr**, so a `.err` log full of `Processed prompts`
  bars is normal output, not a failure — check the SLURM exit code / `sacct`.
- Example smoke test: `scripts/pp_smoke_array.sh` (array, single-GPU ~7-9B models on a
  12-scenario subset under `data/personalprotective_subset/`); `scripts/pp_smoke_large.sh`
  for the ~70B models (4 GPUs each).

**Reasoning models — handled for Qwen3.** Qwen3 / Qwen3.5 are hybrid thinking models
that emit `<think>…</think>` by default, which broke the short-answer MCQ/Likert parsing
(`--max-tokens 5` reads the first `A`/`B` token, but the output started with think
tokens → all `INVALID`). `VLLMClient` now disables thinking for `qwen3*` models by
prefilling the empty think block (`<think>\n\n</think>\n\n`) that the chat template emits
for `enable_thinking=False` — see `disable_thinking` in `VLLMClient.__init__` and
`format_messages_for_qwen` in `src/model_wrappers.py`. We inject it manually because the
prompt string is hand-built (no `apply_chat_template`), and the server-side
`chat_template_kwargs={"enable_thinking": false}` route is unreliable
(vllm-project/vllm#35574). This is scoped to names containing `qwen3`, so Qwen2.5 and
non-Qwen models are untouched. Olmo-3-Instruct, gemma-4, Llama-3.1, and Tulu-3 produce
valid output as-is (no thinking by default). Any future thinking model would need the
same treatment in its `format_messages_for_*`.

Verified on the 12-scenario `personalprotective` subset: Qwen3.5-9B, Llama-3.1-8B,
Tulu-3-8B, Olmo-3-7B, and gemma-4-E4B all load and return valid MCQ + Likert.

## Pipeline

Flow: `generate_scenarios.py` → `filter_scenarios.py` → `evaluate_models.py` → `analyze_experiment.py`.
Example end-to-end scripts live in `scripts/` (`HHH_generation.sh`, `modelspec_generation.sh`, `personalprotective_generation.sh`).

**1. Generate scenarios** (`src/generate_scenarios.py`):
```bash
python src/generate_scenarios.py -v1 helpfulness -v2 honesty -v HHH -n 1200 \
  -m claude-3-5-sonnet-latest -o data/HHH/scenarios -d -dt 0.8 -b 40 --max-tokens 4096
```
`-v1`/`-v2` = the two conflicting values, `-v` = value set, `-n` = count, `-d` = dedup (`-dt` threshold).

**2. Filter scenarios** (`src/filter_scenarios.py`) — validates realism/groundedness, writes a CSV with a `keep_scenario` column:
```bash
python src/filter_scenarios.py -i data/HHH/scenarios/<file>.json -o data/HHH/outputs -m gpt-4.1 -v HHH
```

**3. Evaluate models** (`src/evaluate_models.py`) — two modes: MCQ (default) and interactive (`-i`, simulated user/assistant/judge):
```bash
python src/evaluate_models.py -m claude-3-5-sonnet-latest -d data/HHH -o data/HHH/model_evals --filter
```
Key flags: `-i` interactive, `--filter` use only kept scenarios, `--user-model`/`--judge-model` (default gpt-4o-mini), `--assistant-model` (defaults to `-m`), `--steer-prompt <file>`, `--cache`, `-f` force recompute.

**4. Analyze** (`src/analyze_experiment.py`) — dataset stats + model rankings:
```bash
python src/analyze_experiment.py --model-dir data/HHH --scenario-dir data/HHH \
  --output-dir data/HHH/analysis --value-set HHH --compute-bradley-terry
```
`--wandb` logs to W&B (project default `conflictscope`).

## Value Sets & Data

- Value sets: `value_sets/<NAME>.json` (`HHH`, `modelspec`, `personalprotective`). Define custom sets here and pass the name via `-v`/`--value-set`.
- Data: `data/<value_set>/` holds `prompts.json` and per-model CSVs (e.g. `claude-3-5-sonnet-latest.csv`) for `HHH`, `modelspec`, `personalprotective`.

## Source Modules (`src/`)

- `model_wrappers.py` — unified OpenAI / Anthropic / vLLM interface; add new models here.
- `generate_scenarios.py` — scenario generation with semantic dedup (sentence-transformers).
- `filter_scenarios.py` — scenario validation filter.
- `evaluate_models.py` — MCQ and interactive (simulated-conversation) evaluation.
- `simulated_conversation.py` — multi-turn user/assistant/judge simulation.
- `analyze_experiment.py` — Bradley-Terry rankings and dataset statistics.
- `utils.py` — shared helpers.
