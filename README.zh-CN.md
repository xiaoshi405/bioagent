# BioAgent：蛋白质工具智能体的类型安全执行

**[English](README.md) | 简体中文**

BioAgent 在运行时管理科学对象的身份与来源。请求级 artifact pool 将工具输入绑定到具体源对象；完整计划验证在执行前检查工作流；确定性恢复插入注册表唯一授权的桥接操作，或剪除冗余无效步骤。模型重规划是确定性运行时控制之后的可选补充机制。

本仓库包含蛋白质工具执行层、冻结的 BioToolEval 基准、ReAct、CodeAct、SciToolAgent 和 Biomni-A1 适配器，以及配套论文的离线结果证据包。

[![离线检查](https://github.com/xiaoshi405/bioagent/actions/workflows/offline-checks.yml/badge.svg)](https://github.com/xiaoshi405/bioagent/actions/workflows/offline-checks.yml)

软件版本：**v0.1.0**。详见[安装指南](docs/INSTALLATION.md)、[复现指南](docs/REPRODUCIBILITY.md)和[版本说明](docs/RELEASE_NOTES_v0.1.0.md)。

## 下载并运行离线示例

克隆仓库或解压源码 ZIP，进入仓库根目录后执行命令。

```bash
git clone https://github.com/xiaoshi405/bioagent.git
cd bioagent
```

建议使用 **Python 3.13** 创建全新虚拟环境。这条轻量路径不需要模型密钥、嵌入权重或付费 API 调用。历史实验环境另存于 `results/environment/frozen_external_packages.json`。

```bash
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# Linux/macOS：source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release_assets.py
python scripts/check_environment.py --mode core
python examples/offline_identity.py
python examples/offline_api.py
python scripts/reproduce_paper_results.py --output outputs/reproduced
```

如果 PowerShell 禁止激活脚本，可直接用 `.venv\Scripts\python.exe` 替代上述 `python`，无需修改系统执行策略。

身份示例使用仓库中的多记录 FASTA 测试文件，调用实际的记录选择和 GRAVY 工具。结果复现读取归档的逐次标签及 token 记录，不调用模型，也不产生新的实验结果。

## 工具服务与完整网页 Agent

注册表包含 20 个蛋白质分析工具，覆盖序列、FASTA 记录及集合、PDB 链和比对对象。工具 API 可独立启动：

```bash
python -m uvicorn ToolsAgent.ToolsFuns.Biology.main:app --host 127.0.0.1 --port 8000
```

完整 Agent 和网页界面需要额外依赖。没有 GPU 需求时，可先安装 CPU 版 PyTorch，避免安装 GPU 库：

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-agent.txt
```

首次配置时，将 `example.env` 复制为 `.env`，不要覆盖已有配置。填写自己的 OpenAI 兼容服务 `API_KEY`、`API_BASE` 和 `MODEL_NAME`。后端统一从 `.env` 读取，网页不覆盖这些设置。

检索使用 [BAAI/bge-small-zh](https://huggingface.co/BAAI/bge-small-zh)，与归档模型系列一致。下载完整模型快照，或将 `BGE_MODEL_PATH` 指向已有本地模型：

```bash
hf download BAAI/bge-small-zh --local-dir bge_model
python scripts/check_environment.py --mode agent --before-index
python -m scripts.fix_my_index
python scripts/start_demo.py
```

三个服务全部就绪后，打开 **http://127.0.0.1:8501**。统一启动器管理蛋白质工具 API（8000）、Agent 后端（50005）和 Streamlit 界面（8501），按 Ctrl+C 会停止由它启动的进程。也可以在三个终端中分别启动，见[安装指南](docs/INSTALLATION.md)。

`example.env` 默认开启 artifact binding、validation 和 deterministic repair，并设置随仓库提供的毒性参考 CSV 路径。模型请求会将输入发送到配置的服务商，可能产生 API 费用。嵌入权重下载属于首次配置步骤；离线示例和结果复现均不需要这些权重或密钥。

## 基准与论文证据

BioToolEval 包含 150 道冻结的蛋白质分析题，其中 125 道执行或修复题、25 道拒绝题，附输入文件、gold contract、来源记录、schema 和冻结清单。评价行为见[基准说明](docs/BENCHMARK.md)。

| 研究 | 设计 | 仓库提供的证据 |
|---|---|---|
| 组件消融 | 共享初始计划；7 个条件；2 个模型；3 次重复；6,300 runs | 归档的重复级指标与结果图源数据 |
| 框架比较 | 独立规划；5 种方法；2 个模型；3 次重复；4,500 个计划单元 | 全部 4,500 行逐次分析记录、已记录 token、汇总和配对效应表；历史缺失单元仍明确标记 |
| 受控修复 | 6 个注册修复家族，共 30 个案例 | 冻结套件、运行器、trace schema 和离线回归测试 |

自然恢复与受控桥接套件是两个独立证据来源。主要执行指标为 expected-behavior outcome、execute-task completion 和 safe rejection；对象及输出正确性、回答正确性各有自己的标签和覆盖率。

[复现指南](docs/REPRODUCIBILITY.md)说明证据层级、冻结代码和输入校验、离线表图重建，以及另行版本化的新实验流程。

## 主要目录

```text
app/                          配置、模型与工具接口
ToolsAgent/ToolsFuns/Biology/  蛋白质工具 API 和注册表
scripts/                      artifact 绑定、验证、执行、恢复与评测
scripts/external_baselines/   框架适配器和预算、资源控制
evaluation/biotooleval/        冻结题目、schema、来源记录和输入文件
third_party/                  固定版本的官方 baseline 核心代码与许可证
tests/                        运行时、工具与适配器离线回归测试
examples/                     使用已有测试输入的可运行示例
results/                      归档标签、指标、配图数据与环境记录
```

## 来源、许可与引用

BioAgent 基于 [SciToolAgent](https://github.com/HICAI-ZJU/SciToolAgent) 开发。继承的检索和规划架构，以及新增运行时机制，见 [UPSTREAM.md](UPSTREAM.md)。CodeAct、SciToolAgent 和 Biomni 的官方核心文件固定版本并独立校验，见[适配说明](docs/BASELINES.md)。

根目录 MIT 许可证保留原始版权声明；第三方组件保留各自许可证，包括 Biomni 的 Apache-2.0。生物数据沿用记录的数据库使用条款，见[数据来源](docs/DATA_SOURCES.md)。

软件引用信息见 `CITATION.cff`，其中未将稿件标记为已接收或已发表。打包版本与维护流程见[发布说明](docs/RELEASE.md)和[更新记录](CHANGELOG.md)。
