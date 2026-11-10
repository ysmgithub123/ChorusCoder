# ChorusCoder

[English](README.md) | [中文](README.zh-CN.md)

Code for reproducing the ChorusCoder paper experiments on CrossCodeEval Python and ReccEval Python.

## 1. Environment setup

Use Linux, Python 3.10–3.12, an NVIDIA GPU, and a compatible CUDA driver. Install separate environments for experiments and the vLLM server; encoder comparisons use Transformers 4.x. Run all commands from the repository root. Models are downloaded on first use.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/setup_third_party.py
export CHORUSCODER_PYTHON="$(pwd)/.venv/bin/python"
python3 -m venv .venv-serve
.venv-serve/bin/python -m pip install 'vllm==0.8.5' 'transformers==4.51.3'
```

This server setup follows the [vLLM 0.8.5 installation requirements](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html) and supports NVIDIA GPUs with compute capability ≥7.0.

## 2. Prepare the datasets

**CrossCodeEval:** Obtain the data from the [official project](https://github.com/amazon-science/cceval). After extraction, ensure that `data_sources/crosscodeeval_official/extracted/` contains `python/line_completion.jsonl` and `LICENSES/project_license_map.txt`.

```bash
export CCEVAL_INPUT=data_sources/crosscodeeval_official/extracted/python/line_completion.jsonl
python -m benchmarking.download_cceval_repositories --languages python
python -m benchmarking.prepare_data \
  --input "$CCEVAL_INPUT" --repos-root data_sources/crosscodeeval_repositories \
  --benchmark cceval --language python --output-root data
```

**ReccEval:** Download `metadata.jsonl` and `Source_Code.tar.gz` from the [pinned dataset directory](https://github.com/nju-websoft/DraCo/tree/7d558cd1da2388a7bf1b8b0a30bc4e6fd47cdf83/ReccEval), place them in `data_sources/recceval/`, and run:

```bash
python -m benchmarking.prepare_recceval \
  --recceval-root data_sources/recceval --output-root data
```

The paper uses the complete test sets: 2,665 samples for CrossCodeEval Python and 6,461 for ReccEval. Check the sample counts after preparation and omit `--limit` for full experiments.

## 3. Start the model server

Activate the server environment in a separate terminal and keep the server running:

```bash
source .venv-serve/bin/activate
export GENERATOR_MODEL_PATH=deepseek-ai/deepseek-coder-1.3b-base
bash serve.sh
```

The default port is `8000`. To switch completion models, update both `GENERATOR_MODEL_PATH` for the server and `--model` in the experiment command below.

## 4. Run the experiments

Open another terminal, enter the repository, and activate `.venv`:

```bash
source .venv/bin/activate
export CHORUSCODER_PYTHON="$(pwd)/.venv/bin/python"
python benchmark_cli.py \
  --choruscoder-python "$CHORUSCODER_PYTHON" \
  --method ChorusCoder --model deepseek-ai/deepseek-coder-1.3b-base \
  --benchmark cceval --language python \
  --dataset-root data --output-root benchmark_results/choruscoder \
  --query-cache-dir benchmark_cache/choruscoder
```

By default, the command runs retrieval, generation, and evaluation. Results are organized under `--output-root` by dataset, method, model, and random seed. Use `--benchmark recceval` for ReccEval. Add `--dry-run` to preview the command, or select a stage with `--query-stage prepare|generate|evaluate|all`.

Select paper experiments with `--method`:

- Main experiments: `ChorusCoder`, `InFile-NoRetrieval` (No Retrieval in the paper), `cAST-Prefix100`, `RLCoder`, `CodeRAG-cAST`, `GrepRAG-0.6B`, and `AIRCoder-RRF`.
- Ablation and component comparisons: `MergeQuery`, `Line3Select`, and `Prefix100All`.
- Encoder comparisons: append `-CodeT5p110M` or `-CodeSageSmallV2` to `ChorusCoder` or `cAST-Prefix100`.

Default input/cross-file/generation token budgets are 4096/3072/50, with greedy decoding (temperature=0), batch size=32, and seed=123. The cross-file budget is automatically set to 0 for No Retrieval. The default code encoder is UniXcoder; encoder comparisons use CodeT5+-110M and CodeSage-small-v2.

Run the main experiments on both datasets with `deepseek-ai/deepseek-coder-1.3b-base`, `bigcode/starcoder2-3b`, and `bigcode/starcoder2-7b`. Ablation and encoder comparisons use CCEval Python with DeepSeek.

## 5. AIRCoder and GrepRAG baselines

AIRCoder and GrepRAG require separate environments. GrepRAG also requires the `rg` command (Ubuntu/Debian: `sudo apt install ripgrep`). In the main experiment environment, run:

```bash
for name in aircoder greprag; do
  python3 -m venv ".venv-$name"
  ".venv-$name/bin/python" -m pip install -r "requirements-$name.txt"
done
python -c "from huggingface_hub import snapshot_download; snapshot_download('greprag0/greprag-0.6b', revision='8fd8f3c0ecf6c687daaa69d702742506db74c87f', local_dir='models/greprag-0.6b')"
```

For GrepRAG, use the command in Section 4 with `--method GrepRAG-0.6B` and add `--greprag-python "$(pwd)/.venv-greprag/bin/python"`.

**AIRCoder on a single GPU:** Stop the completion model server, then start Qwen3-8B in the server terminal:

```bash
source .venv-serve/bin/activate
GENERATOR_MODEL_PATH=Qwen/Qwen3-8B MAX_NEW_TOKENS=100 \
  GPU_MEMORY_UTILIZATION=0.70 bash serve.sh
```

Prepare retrieval results in the experiment terminal with `.venv` activated (`--model` always specifies the final completion model):

```bash
python benchmark_cli.py \
  --choruscoder-python "$CHORUSCODER_PYTHON" \
  --aircoder-python "$(pwd)/.venv-aircoder/bin/python" \
  --method AIRCoder-RRF --model deepseek-ai/deepseek-coder-1.3b-base \
  --benchmark cceval --language python \
  --query-stage prepare --output-root benchmark_results/aircoder
```

After preparation, stop the Qwen server and start the corresponding completion model as in Section 3. Keep all other arguments and the output directory unchanged, and rerun the command with `--query-stage generate`, then `--query-stage evaluate`. For ReccEval, follow the same steps with `--benchmark recceval`.

Project code is licensed under [MIT](LICENSE). See [third-party notices](LICENSES/THIRD_PARTY_NOTICES.txt) for third-party licenses and attribution.
