"""
统一配置中心 —— 所有模块通过 Config() 获取配置，配置项只在 .env 文件中修改。

修改记录：
- 2026-09-02：通用无边界序列候选默认最短长度提高到20；短token仅由
  显式序列语义入口接收，降低普通英文术语被误判为蛋白质序列的风险。
- 2026-08-30：新增 Controlled Repair v2 独立数据、Schema、Trace 与结果路径；
  v1 冻结微基准继续保留为默认旧入口，不原地覆盖。
- 2026-08-30：新增baseline Trace扩展路径和L1重规划输出Token上限。
- 2026-08-27：新增正交类型门禁开关和E条件Trace扩展路径；缺省门禁
  延迟继承旧TYPE_PLANNING开关，保持历史配置兼容。
- 2026-08-15：增加类型修复最大剪枝步数配置，限制注册表授权的局部计划改写范围。
- 2026-07-17：增加 LOG_DIR，确保日志路径不依赖进程启动目录。
- 2026-07-18：增加LLM思考开关、温度和请求超时配置，支持固定模型实验。
- 2026-07-19：统一把 .env 中的相对文件路径解析到项目根目录，避免服务
  从不同工作目录启动时找不到 BGE、KG 或本地毒性数据库。
- 2026-07-20：增加 PDB 上传目录配置，为结构文件工具提供统一受控路径。
- 2026-07-24：增加 Protein FASTA 上传大小配置。
- 2026-07-25：增加类型规划与保守修复开关，支持v0.2独立适配和后续消融实验。
- 2026-07-26：类型规划开关接入v0.3b主链路；默认仍关闭以保持baseline。
- 2026-07-28：增加v0.3c实验Trace Schema和本地结果目录配置。
- 2026-07-30：增加v0.3d题目Schema、离线dry-run数据集和单题硬超时配置。
- 2026-07-30：增加v0.3e人工复核盲化密钥配置；密钥只从.env读取。
- 2026-08-11：增加正式实验Python版本和双服务预检超时配置。
- 2026-08-12：增加序列候选定位阈值配置，区分通用候选与显式短序列。
- 2026-08-13：增加Controlled Repair Track数据、Schema、结果目录和单元超时配置。
- 2026-08-14：增加Topology v2-lite多记录FASTA、蛋白质比对和多跳类型
  修复边界配置，避免在领域工具与修复器中硬编码实验参数。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 修改记录：2026-08-11，新增可配置的 llama-index 本地缓存目录，避免依赖用户级缓存路径。

# 自动加载项目根目录下的 .env 文件
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
_project_root = _env_path.parent
load_dotenv(_env_path, override=True)


def _resolve_project_path(value: str, allow_empty: bool = False) -> str:
    """把配置路径稳定解析为绝对路径；可选配置允许保留空字符串。"""
    try:
        configured = str(value or "").strip()
        if not configured:
            return "" if allow_empty else str(_project_root)
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = _project_root / path
        return str(path.resolve())
    except Exception as exc:
        raise RuntimeError(f"解析项目配置路径失败: {value!r}: {exc}") from exc


def _read_bool_env(name: str, default: bool = False) -> bool:
    """从环境变量读取布尔开关；非法或缺失值安全回退到默认值。"""
    try:
        fallback = "true" if default else "false"
        return os.getenv(name, fallback).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    except Exception:
        return bool(default)


class Config:
    """单例式配置读取器：所有字段从 .env 统一读取，切换模型/API 只改 .env 即可。"""

    def __init__(self):
        # ---- 1. API 配置 ----
        self.API_BASE = os.getenv("API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.API_KEY = os.getenv("API_KEY", "")

        # 兼容旧代码中使用 OPENAI_* / DASHSCOPE_* 名称的地方
        self.OPENAI_API_BASE = self.API_BASE
        self.OPENAI_API_KEY = self.API_KEY
        self.DASHSCOPE_API_BASE = self.API_BASE
        self.DASHSCOPE_API_KEY = self.API_KEY

        # ---- 2. 模型选型 ----
        self.MODEL_NAME = os.getenv("MODEL_NAME", "").strip()
        self.LLM_ENABLE_THINKING = _read_bool_env("LLM_ENABLE_THINKING")
        try:
            self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
        except (TypeError, ValueError):
            self.LLM_TEMPERATURE = 0.0
        try:
            self.LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        except (TypeError, ValueError):
            self.LLM_TIMEOUT_SECONDS = 60.0
        self.LLAMAINDEX_FAKE_MODEL = os.getenv("LLAMAINDEX_FAKE_MODEL", "gpt-3.5-turbo")

        # ---- 2.1 类型约束规划实验开关 ----
        # v0.3b仍默认关闭；只有实验配置显式开启时才切换类型化执行分支。
        self.TYPE_PLANNING_ENABLED = _read_bool_env("TYPE_PLANNING_ENABLED")
        # 2026-08-27：把类型化运行时、类型门禁和确定性修复拆成三个正交开关。
        # 环境变量缺省时继承旧TYPE_PLANNING语义，保证历史B0/B1/B2配置不变。
        validation_raw = os.getenv("TYPE_VALIDATION_ENABLED")
        self.TYPE_VALIDATION_ENABLED = (
            None
            if validation_raw is None
            else validation_raw.strip().lower() in {"1", "true", "yes", "on"}
        )
        self.TYPE_REPAIR_ENABLED = _read_bool_env("TYPE_REPAIR_ENABLED")
        try:
            self.TYPE_REPAIR_MAX_BRIDGE_HOPS = min(
                4,
                max(1, int(os.getenv("TYPE_REPAIR_MAX_BRIDGE_HOPS", "2"))),
            )
        except (TypeError, ValueError):
            self.TYPE_REPAIR_MAX_BRIDGE_HOPS = 2
        try:
            self.TYPE_REPAIR_MAX_PRUNED_STEPS = min(
                4,
                max(2, int(os.getenv("TYPE_REPAIR_MAX_PRUNED_STEPS", "2"))),
            )
        except (TypeError, ValueError):
            self.TYPE_REPAIR_MAX_PRUNED_STEPS = 2
        # ---- 2.2 序列候选定位阈值 ----
        # 定位只决定“是否存在候选”，字符合法性由初始Artifact校验层统一负责。
        try:
            self.SEQUENCE_CANDIDATE_MIN_LENGTH = max(
                1,
                int(os.getenv("SEQUENCE_CANDIDATE_MIN_LENGTH", "20")),
            )
        except (TypeError, ValueError):
            self.SEQUENCE_CANDIDATE_MIN_LENGTH = 20
        try:
            self.SEQUENCE_CANDIDATE_MIN_VALID_RESIDUES = max(
                1,
                int(
                    os.getenv(
                        "SEQUENCE_CANDIDATE_MIN_VALID_RESIDUES",
                        "8",
                    )
                ),
            )
        except (TypeError, ValueError):
            self.SEQUENCE_CANDIDATE_MIN_VALID_RESIDUES = 8
        try:
            configured_ratio = float(
                os.getenv("SEQUENCE_CANDIDATE_MIN_VALID_RATIO", "0.6")
            )
            self.SEQUENCE_CANDIDATE_MIN_VALID_RATIO = min(
                1.0,
                max(0.0, configured_ratio),
            )
        except (TypeError, ValueError):
            self.SEQUENCE_CANDIDATE_MIN_VALID_RATIO = 0.6
        try:
            self.EXPLICIT_SEQUENCE_CANDIDATE_MIN_LENGTH = max(
                1,
                int(
                    os.getenv(
                        "EXPLICIT_SEQUENCE_CANDIDATE_MIN_LENGTH",
                        "1",
                    )
                ),
            )
        except (TypeError, ValueError):
            self.EXPLICIT_SEQUENCE_CANDIDATE_MIN_LENGTH = 1

        # ---- 3. Embedding 配置 ----
        self.BGE_MODEL_PATH = _resolve_project_path(
            os.getenv("BGE_MODEL_PATH", "bge_model")
        )
        self.LLAMA_INDEX_CACHE_DIR = _resolve_project_path(
            os.getenv("LLAMA_INDEX_CACHE_DIR", ".cache/llama_index")
        )
        # 保留旧字段名兼容
        self.EMBEDDING_MODEL_NAME = self.BGE_MODEL_PATH

        # ---- 4. 路径配置 ----
        self.PERSIST_DIR = _resolve_project_path(os.getenv("PERSIST_DIR", "storage"))
        self.DATA_FILE_PATH = _resolve_project_path(
            os.getenv("DATA_FILE_PATH", "data/toolsKG_full.xlsx")
        )
        self.LOG_DIR = _resolve_project_path(os.getenv("LOG_DIR", "logs"))
        self.UPLOAD_DIR = _resolve_project_path(os.getenv("UPLOAD_DIR", "uploads"))
        self.EXPERIMENT_RESULTS_DIR = _resolve_project_path(
            os.getenv("EXPERIMENT_RESULTS_DIR", "evaluation/results")
        )
        self.EXPERIMENT_TRACE_SCHEMA_PATH = _resolve_project_path(
            os.getenv(
                "EXPERIMENT_TRACE_SCHEMA_PATH",
                "evaluation/biotooleval/experiment_trace_schema_v1.json",
            )
        )
        self.EXPERIMENT_TRACE_EXTENSION_PATH = _resolve_project_path(
            os.getenv(
                "EXPERIMENT_TRACE_EXTENSION_PATH",
                "evaluation/biotooleval/experiment_trace_schema_e_v1.json",
            )
        )
        self.EXPERIMENT_TRACE_BASELINE_EXTENSION_PATH = _resolve_project_path(
            os.getenv(
                "EXPERIMENT_TRACE_BASELINE_EXTENSION_PATH",
                "evaluation/biotooleval/experiment_trace_schema_baseline_v1.json",
            )
        )
        self.EXPERIMENT_TRACE_HYBRID_EXTENSION_PATH = _resolve_project_path(
            os.getenv(
                "EXPERIMENT_TRACE_HYBRID_EXTENSION_PATH",
                "evaluation/biotooleval/experiment_trace_schema_hybrid_v1.json",
            )
        )
        self.EXPERIMENT_QUESTION_SCHEMA_PATH = _resolve_project_path(
            os.getenv(
                "EXPERIMENT_QUESTION_SCHEMA_PATH",
                "evaluation/biotooleval/question_schema_v1.json",
            )
        )
        self.EXPERIMENT_DRY_RUN_DATASET_PATH = _resolve_project_path(
            os.getenv(
                "EXPERIMENT_DRY_RUN_DATASET_PATH",
                "tests/fixtures/biotooleval_dry_run.json",
            )
        )
        self.CONTROLLED_REPAIR_DATASET_PATH = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_DATASET_PATH",
                "evaluation/biotooleval/controlled_repair_v1.json",
            )
        )
        self.CONTROLLED_REPAIR_DATASET_SCHEMA_PATH = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_DATASET_SCHEMA_PATH",
                "evaluation/biotooleval/controlled_repair_dataset_schema_v1.json",
            )
        )
        self.CONTROLLED_REPAIR_TRACE_SCHEMA_PATH = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_TRACE_SCHEMA_PATH",
                "evaluation/biotooleval/controlled_repair_trace_schema_v1.json",
            )
        )
        self.CONTROLLED_REPAIR_RESULTS_DIR = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_RESULTS_DIR",
                "evaluation/results/controlled_repair",
            )
        )
        self.CONTROLLED_REPAIR_V2_DATASET_PATH = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_V2_DATASET_PATH",
                "evaluation/biotooleval/controlled_repair_v2.json",
            )
        )
        self.CONTROLLED_REPAIR_V2_DATASET_SCHEMA_PATH = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_V2_DATASET_SCHEMA_PATH",
                "evaluation/biotooleval/controlled_repair_dataset_schema_v2.json",
            )
        )
        self.CONTROLLED_REPAIR_V2_TRACE_SCHEMA_PATH = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_V2_TRACE_SCHEMA_PATH",
                "evaluation/biotooleval/controlled_repair_trace_schema_v2.json",
            )
        )
        self.CONTROLLED_REPAIR_V2_RESULTS_DIR = _resolve_project_path(
            os.getenv(
                "CONTROLLED_REPAIR_V2_RESULTS_DIR",
                "evaluation/results/controlled_repair_v2",
            )
        )
        self.EXPERIMENT_REVIEW_BLINDING_KEY = os.getenv(
            "EXPERIMENT_REVIEW_BLINDING_KEY",
            "",
        ).strip()
        self.EXPERIMENT_PYTHON_VERSION = os.getenv(
            "EXPERIMENT_PYTHON_VERSION",
            "",
        ).strip()
        try:
            configured_preflight_timeout = float(
                os.getenv("EXPERIMENT_PREFLIGHT_TIMEOUT_SECONDS", "60")
            )
            self.EXPERIMENT_PREFLIGHT_TIMEOUT_SECONDS = min(
                60.0,
                max(1.0, configured_preflight_timeout),
            )
        except (TypeError, ValueError):
            self.EXPERIMENT_PREFLIGHT_TIMEOUT_SECONDS = 60.0
        try:
            configured_experiment_timeout = float(
                os.getenv("EXPERIMENT_QUESTION_TIMEOUT_SECONDS", "120")
            )
            self.EXPERIMENT_QUESTION_TIMEOUT_SECONDS = min(
                1800.0,
                max(1.0, configured_experiment_timeout),
            )
        except (TypeError, ValueError):
            self.EXPERIMENT_QUESTION_TIMEOUT_SECONDS = 120.0
        # 修改记录：2026-08-15，实验规划只允许基础设施故障重试；模型内容响应不重抽。
        try:
            configured_plan_network_attempts = int(
                os.getenv("EXPERIMENT_PLAN_NETWORK_MAX_ATTEMPTS", "3")
            )
            self.EXPERIMENT_PLAN_NETWORK_MAX_ATTEMPTS = min(
                5,
                max(1, configured_plan_network_attempts),
            )
        except (TypeError, ValueError):
            self.EXPERIMENT_PLAN_NETWORK_MAX_ATTEMPTS = 3
        try:
            configured_plan_backoff = float(
                os.getenv("EXPERIMENT_PLAN_NETWORK_BACKOFF_SECONDS", "1")
            )
            self.EXPERIMENT_PLAN_NETWORK_BACKOFF_SECONDS = min(
                30.0,
                max(0.0, configured_plan_backoff),
            )
        except (TypeError, ValueError):
            self.EXPERIMENT_PLAN_NETWORK_BACKOFF_SECONDS = 1.0
        try:
            configured_replan_tokens = int(
                os.getenv("EXPERIMENT_LLM_REPLAN_MAX_OUTPUT_TOKENS", "256")
            )
            self.EXPERIMENT_LLM_REPLAN_MAX_OUTPUT_TOKENS = min(
                1024,
                max(32, configured_replan_tokens),
            )
        except (TypeError, ValueError):
            self.EXPERIMENT_LLM_REPLAN_MAX_OUTPUT_TOKENS = 256
        try:
            configured_controlled_timeout = float(
                os.getenv("CONTROLLED_REPAIR_TIMEOUT_SECONDS", "30")
            )
            self.CONTROLLED_REPAIR_TIMEOUT_SECONDS = min(
                300.0,
                max(1.0, configured_controlled_timeout),
            )
        except (TypeError, ValueError):
            self.CONTROLLED_REPAIR_TIMEOUT_SECONDS = 30.0
        try:
            configured_upload_limit = int(os.getenv("MAX_PDB_UPLOAD_MB", "10"))
            self.MAX_PDB_UPLOAD_MB = min(100, max(1, configured_upload_limit))
        except (TypeError, ValueError):
            self.MAX_PDB_UPLOAD_MB = 10
        try:
            configured_fasta_limit = int(os.getenv("MAX_FASTA_UPLOAD_MB", "5"))
            self.MAX_FASTA_UPLOAD_MB = min(100, max(1, configured_fasta_limit))
        except (TypeError, ValueError):
            self.MAX_FASTA_UPLOAD_MB = 5
        try:
            configured_record_limit = int(
                os.getenv("MAX_MULTI_FASTA_RECORDS", "1000")
            )
            self.MAX_MULTI_FASTA_RECORDS = min(
                100000,
                max(2, configured_record_limit),
            )
        except (TypeError, ValueError):
            self.MAX_MULTI_FASTA_RECORDS = 1000
        try:
            configured_alignment_limit = int(
                os.getenv("MAX_PROTEIN_ALIGNMENT_SEQUENCE_LENGTH", "5000")
            )
            self.MAX_PROTEIN_ALIGNMENT_SEQUENCE_LENGTH = min(
                100000,
                max(1, configured_alignment_limit),
            )
        except (TypeError, ValueError):
            self.MAX_PROTEIN_ALIGNMENT_SEQUENCE_LENGTH = 5000
        self.PROTEIN_ALIGNMENT_MATRIX = os.getenv(
            "PROTEIN_ALIGNMENT_MATRIX",
            "BLOSUM62",
        ).strip() or "BLOSUM62"
        try:
            self.PROTEIN_ALIGNMENT_OPEN_GAP_SCORE = min(
                0.0,
                float(os.getenv("PROTEIN_ALIGNMENT_OPEN_GAP_SCORE", "-10")),
            )
        except (TypeError, ValueError):
            self.PROTEIN_ALIGNMENT_OPEN_GAP_SCORE = -10.0
        try:
            self.PROTEIN_ALIGNMENT_EXTEND_GAP_SCORE = min(
                0.0,
                float(os.getenv("PROTEIN_ALIGNMENT_EXTEND_GAP_SCORE", "-0.5")),
            )
        except (TypeError, ValueError):
            self.PROTEIN_ALIGNMENT_EXTEND_GAP_SCORE = -0.5
        self.TOXIN_COMPOUND = _resolve_project_path(
            os.getenv("TOXIN_COMPOUND", ""), allow_empty=True
        )
        self.TOXIN_PROTEIN = _resolve_project_path(
            os.getenv("TOXIN_PROTEIN", ""), allow_empty=True
        )

        # ---- 5. 服务配置 ----
        self.SERVER_PORT = int(os.getenv("SERVER_PORT", "50005"))
        self.AGENT_API_BASE = os.getenv(
            "AGENT_API_BASE",
            f"http://127.0.0.1:{self.SERVER_PORT}",
        ).strip()
        self.HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

        # Biology FastAPI 子服务地址（默认 :8000）
        self.BIO_API_PORT = int(os.getenv("BIO_API_PORT", "8000"))
        self.BIO_API_BASE = os.getenv("BIO_API_BASE", f"http://127.0.0.1:{self.BIO_API_PORT}")
