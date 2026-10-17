"""Load application settings from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv


_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
_project_root = _env_path.parent
load_dotenv(_env_path, override=True)


def _resolve_project_path(value: str, allow_empty: bool = False) -> str:
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
    """Application settings loaded from .env."""

    def __init__(self):
        self.API_BASE = os.getenv("API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.API_KEY = os.getenv("API_KEY", "")

        # Keep legacy API setting aliases.
        self.OPENAI_API_BASE = self.API_BASE
        self.OPENAI_API_KEY = self.API_KEY
        self.DASHSCOPE_API_BASE = self.API_BASE
        self.DASHSCOPE_API_KEY = self.API_KEY

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

        self.TYPE_PLANNING_ENABLED = _read_bool_env("TYPE_PLANNING_ENABLED")
        # An unset validation flag inherits TYPE_PLANNING_ENABLED.
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
        # Candidate extraction does not validate amino acid content.
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

        self.BGE_MODEL_PATH = _resolve_project_path(
            os.getenv("BGE_MODEL_PATH", "bge_model")
        )
        self.LLAMA_INDEX_CACHE_DIR = _resolve_project_path(
            os.getenv("LLAMA_INDEX_CACHE_DIR", ".cache/llama_index")
        )
        self.EMBEDDING_MODEL_NAME = self.BGE_MODEL_PATH

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

        self.SERVER_PORT = int(os.getenv("SERVER_PORT", "50005"))
        self.AGENT_API_BASE = os.getenv(
            "AGENT_API_BASE",
            f"http://127.0.0.1:{self.SERVER_PORT}",
        ).strip()
        self.HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

        self.BIO_API_PORT = int(os.getenv("BIO_API_PORT", "8000"))
        self.BIO_API_BASE = os.getenv("BIO_API_BASE", f"http://127.0.0.1:{self.BIO_API_PORT}")
