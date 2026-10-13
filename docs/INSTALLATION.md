# Installation / 安装与启动

Run every command from the repository root. The supported, verified setup uses Python 3.13 and a fresh virtual environment. See [English README](../README.md) or [中文 README](../README.zh-CN.md) for the project overview.

## 1. Lightweight offline environment / 轻量离线环境

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. If PowerShell blocks activation, use `.venv\Scripts\python.exe` directly instead of `python`; no execution-policy change is needed.

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py --mode core
python scripts/verify_release_assets.py
python scripts/run_release_tests.py --group core
python examples/offline_identity.py
python examples/offline_api.py
python scripts/reproduce_paper_results.py --output outputs/reproduced
```

This path needs no API key, embedding model, background service or model request. The examples use actual bundled FASTA fixtures and registered tools. Result reconstruction reads archived labels and recorded usage. / 此路径无需密钥、嵌入模型或后台服务；示例使用已有 FASTA 文件和真实工具，结果重建读取归档记录。

## 2. Standalone protein tool API / 独立工具 API

```bash
python -m uvicorn ToolsAgent.ToolsFuns.Biology.main:app --host 127.0.0.1 --port 8000
```

API documentation: **http://127.0.0.1:8000/docs**. To use the bundled toxicity reference with this service, copy `example.env` to `.env` first. Its `TOXIN_PROTEIN=data/toxin_protein.csv` setting enables the existing local similarity tool. / 工具文档见该地址；毒性相似性工具需要示例配置中的 CSV 路径。

## 3. Full agent and web interface / 完整 Agent 与网页

Install optional dependencies. The following CPU wheel is suitable for the local BGE retrieval path; GPU users can select a wheel using the [official PyTorch installer](https://pytorch.org/get-started/locally/).

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-agent.txt
python -m pip check
```

**First setup only:** copy `example.env` to `.env`. Do not overwrite a working configuration. / 仅在首次配置时复制，避免覆盖已有密钥。

```bash
python -c "import shutil; shutil.copyfile('example.env', '.env')"
```

Edit `.env`:

| Setting / 配置项 | Value to supply / 内容 |
|---|---|
| `API_KEY` | Your own provider key / 自己的服务商密钥 |
| `API_BASE` | OpenAI-compatible provider base URL / 兼容服务根地址 |
| `MODEL_NAME` | An available model name for that account / 账号实际可用的模型名 |
| `BGE_MODEL_PATH` | Complete local embedding snapshot; default `bge_model` / 完整本地嵌入模型目录 |
| `TOXIN_PROTEIN` | Keep `data/toxin_protein.csv` for the bundled reference / 随仓库提供的数据库 |

The backend reads API settings from `.env`; browser controls do not supply or override credentials. `example.env` enables typed planning, validation and deterministic repair. Changing these flags changes the execution condition. / API 配置由后端统一读取；示例配置默认开启论文的确定性运行时机制。

Download the complete [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh) snapshot, including configuration, tokenizer, pooling and weights. The CLI is installed with the optional dependencies. You may instead point `BGE_MODEL_PATH` to an existing complete local copy. Downloaded snapshots are local assets, not part of the source release or a substitute for historical hash verification.

```bash
hf download BAAI/bge-small-zh --local-dir bge_model
python scripts/check_environment.py --mode agent --before-index
python -m scripts.fix_my_index
python scripts/check_environment.py --mode agent
python scripts/start_demo.py
```

The index builder uses the local embedding model and registry with a local MockLLM; it does not call a model provider. After startup, open **http://127.0.0.1:8501**. Press Ctrl+C to stop the launcher's own services. / 索引构建不调用云端模型；所有服务就绪后打开网页，Ctrl+C 停止启动器管理的进程。

To start separately, use three terminals with the **same virtual environment**:

```bash
# Terminal 1: protein tools / 蛋白质工具
python -m uvicorn ToolsAgent.ToolsFuns.Biology.main:app --host 127.0.0.1 --port 8000
# Terminal 2: agent backend / Agent 后端
python scripts/start_agent.py --host 127.0.0.1 --port 50005
# Terminal 3: web interface / 网页界面
python -m streamlit run web_demo.py --server.address 127.0.0.1 --server.port 8501
```

The agent launcher explicitly loads `app.py` because `app/` is also a Python package; `uvicorn app:app` resolves the wrong entry point. `python scripts/start_agent.py --check-only` checks ASGI loading without starting a server or calling a model. / 不要使用 `uvicorn app:app`，该名称会与目录冲突。

Online agent requests send protein inputs and tool returns to the configured provider and can incur charges. The startup checks, offline examples and result reconstruction do not. / 在线问答会发送输入并可能产生费用；启动检查与离线运行不会。

## 4. Baseline environment / Baseline 环境

```bash
python -m pip install -r requirements-baselines.txt
python -m pip check
python scripts/check_environment.py --mode baselines
python scripts/run_release_tests.py --group all
```

Released tests use local transports and scripted test responses, not paid model requests. The full historical model/tool events are not bundled; the result package supports archived table/plot reconstruction. The recorded experiment environment remains in `results/environment/frozen_external_packages.json`. New online experiments require a separately versioned preparation/freeze; do not overwrite the archived freeze. / 此处是离线软件回归验证；新在线实验须另行冻结，不覆盖历史实验记录。

## Troubleshooting / 常见问题

- **Missing dependencies:** use the same virtual environment for installation and startup; `python -m pip check` checks dependency conflicts. / 检查安装与启动是否使用同一解释器。
- **Missing local BGE model:** download the whole snapshot and check `BGE_MODEL_PATH`; the index builder deliberately runs offline. / 必须是完整模型目录，不能仅复制权重。
- **Missing index:** run `python -m scripts.fix_my_index` after preparing weights. / 准备模型后再构建索引。
- **Occupied port:** stop your own previous service, or configure different ports consistently. The demo launcher will not stop unrelated processes. / 启动器不会结束其他程序。
- **Backend error in the web interface:** its message is displayed directly; inspect local `logs/` for details. / 网页显示后端错误，详细记录保存在本地日志。
