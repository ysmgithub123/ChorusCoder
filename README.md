# ChorusCoder

ChorusCoder 论文实验代码，支持 CrossCodeEval Python 和 ReccEval Python。

## 1. 安装环境

使用 Linux、Python 3.10–3.12、NVIDIA GPU 和匹配的 CUDA 驱动。实验环境与 vLLM 服务环境分开安装；编码器适配使用 Transformers 4.x。以下命令均在仓库根目录执行，首次运行需要下载模型。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/setup_third_party.py
export CHORUSCODER_PYTHON="$(pwd)/.venv/bin/python"
python3 -m venv .venv-serve
.venv-serve/bin/python -m pip install 'vllm==0.8.5' 'transformers==4.51.3'
```

服务环境示例按 [vLLM 0.8.5 的安装要求](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html) 配置，支持计算能力 ≥7.0 的 NVIDIA GPU。

## 2. 准备数据

**CrossCodeEval：** 从 [官方项目](https://github.com/amazon-science/cceval) 获取数据，解压后确保 `data_sources/crosscodeeval_official/extracted/` 下包含 `python/line_completion.jsonl` 和 `LICENSES/project_license_map.txt`。

```bash
export CCEVAL_INPUT=data_sources/crosscodeeval_official/extracted/python/line_completion.jsonl
python -m benchmarking.download_cceval_repositories --languages python
python -m benchmarking.prepare_data \
  --input "$CCEVAL_INPUT" --repos-root data_sources/crosscodeeval_repositories \
  --benchmark cceval --language python --output-root data
```

**ReccEval：** 从 [固定版本数据目录](https://github.com/nju-websoft/DraCo/tree/7d558cd1da2388a7bf1b8b0a30bc4e6fd47cdf83/ReccEval) 下载 `metadata.jsonl` 和 `Source_Code.tar.gz`，放入 `data_sources/recceval/`，然后执行：

```bash
python -m benchmarking.prepare_recceval \
  --recceval-root data_sources/recceval --output-root data
```

论文使用完整测试集：CrossCodeEval Python 为 2,665 条，ReccEval 为 6,461 条。核对准备输出的样本数，完整实验不设置 `--limit`。

## 3. 启动模型服务

在单独终端中激活服务环境，运行并保持服务开启：

```bash
source .venv-serve/bin/activate
export GENERATOR_MODEL_PATH=deepseek-ai/deepseek-coder-1.3b-base
bash serve.sh
```

默认端口为 `8000`。更换补全模型时，同时修改服务的 `GENERATOR_MODEL_PATH` 和下方实验命令的 `--model`。

## 4. 运行实验

另开终端，进入仓库并激活 `.venv`：

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

默认完成检索、生成和评价，结果在 `--output-root` 下按数据集、方法、模型和随机种子分目录保存。运行 ReccEval 时改为 `--benchmark recceval`。追加 `--dry-run` 可预览命令；`--query-stage prepare|generate|evaluate|all` 可分阶段执行。

通过 `--method` 切换论文实验：

- 主实验：`ChorusCoder`、`InFile-NoRetrieval`（论文中的 No Retrieval）、`cAST-Prefix100`、`RLCoder`、`CodeRAG-cAST`、`GrepRAG-0.6B`、`AIRCoder-RRF`。
- 消融／组合对照：`MergeQuery`、`Line3Select`、`Prefix100All`。
- 编码器对照：在 `ChorusCoder` 或 `cAST-Prefix100` 后加 `-CodeT5p110M` 或 `-CodeSageSmallV2`。

默认输入／跨文件／生成预算为 4096／3072／50，贪心解码（temperature=0）、batch size=32、seed=123；No Retrieval 的跨文件预算自动设为 0。默认代码编码器为 UniXcoder，编码器对照分别使用 CodeT5+-110M 和 CodeSage-small-v2。

主实验分别使用 `deepseek-ai/deepseek-coder-1.3b-base`、`bigcode/starcoder2-3b`、`bigcode/starcoder2-7b`，在两个数据集上运行；消融和编码器对照使用 CCEval Python 与 DeepSeek。

## 5. AIRCoder 与 GrepRAG 对照

AIRCoder 和 GrepRAG 需要独立环境。GrepRAG 还需要系统命令 `rg`（Ubuntu/Debian：`sudo apt install ripgrep`）。在主环境终端中准备：

```bash
for name in aircoder greprag; do
  python3 -m venv ".venv-$name"
  ".venv-$name/bin/python" -m pip install -r "requirements-$name.txt"
done
python -c "from huggingface_hub import snapshot_download; snapshot_download('greprag0/greprag-0.6b', revision='8fd8f3c0ecf6c687daaa69d702742506db74c87f', local_dir='models/greprag-0.6b')"
```

GrepRAG 沿用第 4 节的命令，将方法改为 `GrepRAG-0.6B`，追加 `--greprag-python "$(pwd)/.venv-greprag/bin/python"`。

**单卡 AIRCoder：** 先停止补全模型服务，在服务终端启动 Qwen3-8B：

```bash
source .venv-serve/bin/activate
GENERATOR_MODEL_PATH=Qwen/Qwen3-8B MAX_NEW_TOKENS=100 \
  GPU_MEMORY_UTILIZATION=0.70 bash serve.sh
```

在已激活 `.venv` 的实验终端准备检索结果（`--model` 始终指定最终补全模型）：

```bash
python benchmark_cli.py \
  --choruscoder-python "$CHORUSCODER_PYTHON" \
  --aircoder-python "$(pwd)/.venv-aircoder/bin/python" \
  --method AIRCoder-RRF --model deepseek-ai/deepseek-coder-1.3b-base \
  --benchmark cceval --language python \
  --query-stage prepare --output-root benchmark_results/aircoder
```

准备完成后，停止 Qwen 服务，按第 3 节启动对应补全模型。保持上述命令的其他参数与输出目录一致，依次将 `--query-stage` 改为 `generate` 和 `evaluate` 执行。ReccEval 使用相同步骤，将 `--benchmark` 改为 `recceval`。

项目自有代码采用 [MIT](LICENSE)；第三方许可与署名见 [第三方声明](LICENSES/THIRD_PARTY_NOTICES.txt)。
