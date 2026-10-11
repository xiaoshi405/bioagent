# -*- coding: utf-8 -*-
"""
app/tools/tool_runner.py
Agent 工具执行出口 —— 将 KG 规划出的工具名转发给 Biology FastAPI 服务 (:8000)

修改记录
1. 新增 TOOL_NAME_TO_API 映射表，解决 KG 工具名与 B 接口路径不一致问题
2. 新增 extract_sequence() 函数，从用户自然语言中提取氨基酸序列，
   避免把整句话传给 sequence 字段导致计算失败
3. 全面包裹 try-except + 超时控制，B 服务未启动或网络异常时
   返回标准 error JSON，主进程不崩溃
4. 2026-07-19：注册六个新增序列工具；统一返回合法 JSON；未知工具不再
   静默兜底到综合分析接口，避免执行错误功能。
5. 2026-07-20：增加PDB文件工具分流，使用file_path_list构造请求，避免把
   文件问题错误地按氨基酸序列解析。
6. 2026-07-21：修复疑似序列中的数字/分隔符被静默删除的问题；Runner只
   负责定位候选token，保留非法字符交给Biology服务统一拒绝。
7. 2026-07-24：增加单记录Protein FASTA文件分流和解析接口映射。
8. 2026-07-24：增加显式PDB链选择分流；缺少chain_id时拒绝调用。
9. 2026-07-24：PDB链ID按用户原样传递，并兼容chain_id=B等显式写法；
10. 2026-08-09：补充“A链/A 链”等前置链ID中文写法，消除链参数抽取的
   空格与语序敏感性；仍只接受用户显式给出的单字符链ID。
   禁止执行层擅自转换大小写，避免选中错误链或把有效小写链改成无效链。
10. 2026-07-26：增加typed_inputs精确输入通道，使v0.3a执行器可直接
    使用Artifact Pool真实对象，避免再次让LLM或正则猜测前序输出。
11. 2026-07-26：拆出严格的extract_sequence_candidate()，供v0.3b在
    请求开始时识别初始序列对象；旧extract_sequence()保留原有回退行为。
12. 2026-08-11：工具别名规范化同时忽略大小写、下划线、连字符和点号，
    供规划前显式工具注册门禁复用。
13. 2026-08-12：将序列候选定位与字符合法性校验分离；仅对带序列标签、
    引号或代码边界的短token放宽定位，并排除PDB ID与chain ID上下文。
14. 2026-08-14：接入Topology v2-lite的多记录FASTA与比对对象映射；
    类型化通道按精确类型构造结构化JSON，baseline通道保守传递前序对象。
15. 2026-09-02：结构化序列代码块优先于题面标识符；排除UniProt accession，
    并收紧非显式非法候选，避免数据库名称遮蔽真实序列或上传文件。
16. 2026-09-04：普通与类型化执行共用请求级工具参数上下文；ANM显式
    n_modes、cutoff和chain_id不再因Artifact Pool执行而丢失。
"""

#执行层（executor），请求生物计算的Fastapi服务的端口
import json
import re
from pathlib import Path
import aiohttp
from app.core.config import Config
from app.core.logger import get_logger
from app.tools.fasta_input_parser import extract_fasta_record_identifiers
from app.tools.tool_parameters import (
    build_tool_parameter_context,
    extract_pdb_options,
    resolve_tool_parameters,
    validate_tool_parameters,
)

_cfg = Config()
logger = get_logger("AgentService", "agent.log")


# ------------------------------------------------------------------ #
#  1. KG工具名  →  B后端接口路径  映射表
#     KG检索返回的是 tools_dict.py 里的原始名（大驼峰）
#     B的接口路径是小写+下划线，两者必须在这里对齐
#     后续B新增接口时，在此表里补一行即可，无需改其他代码
# ------------------------------------------------------------------ #
TOOL_NAME_TO_API: dict[str, str] = {
    # ---------- 核心计算接口 ----------
    "ComputePiMw":                    "/api/compute_pi_mw",
    "compute_pI_mw":                  "/api/compute_pi_mw",
    "ProteinIsoelectricPointCalculator": "/api/compute_pi_mw",

    # ---------- 本地序列描述 ----------
    "compute_gravy":                   "/api/compute_gravy",
    "ComputeGRAVY":                    "/api/compute_gravy",
    "compute_secondary_structure_fraction": "/api/compute_secondary_structure_fraction",
    "ComputeSecondaryStructureFraction": "/api/compute_secondary_structure_fraction",
    "compute_amino_acid_composition":  "/api/compute_amino_acid_composition",
    "ComputeAminoAcidComposition":     "/api/compute_amino_acid_composition",
    "compute_instability_index":       "/api/compute_instability_index",
    "ComputeInstabilityIndex":         "/api/compute_instability_index",
    "compute_extinction_coefficient":  "/api/compute_extinction_coefficient",
    "ComputeExtinctionCoefficient":    "/api/compute_extinction_coefficient",

    # ---------- 信号肽预测 ----------
    "PredictSignalPeptide":           "/api/predict_signalpeptide",
    "predict_signalpeptide":          "/api/predict_signalpeptide",

    # ---------- 跨膜区预测 ----------
    "PredictTransmembraneRegion":     "/api/predict_transmembrane",
    "predict_transmembrane":          "/api/predict_transmembrane",

    # ---------- 溶解度预测 ----------
    "ProteinSolubilityPredictor":     "/api/predict_solubility",
    "PredictProteinSolubility":       "/api/predict_solubility",
    "predict_solubility":             "/api/predict_solubility",

    # ---------- 本地毒性蛋白相似性筛查 ----------
    "check_protein_toxicity":         "/api/check_protein_toxicity",
    "CheckProteinToxicity":           "/api/check_protein_toxicity",

    # ---------- 本地PDB结构分析 ----------
    "analyze_pdb_structure":          "/api/analyze_pdb_structure",
    "AnalyzeProteinStructure":        "/api/analyze_pdb_structure",
    "extract_pdb_sequences":          "/api/extract_pdb_sequences",
    "ExtractPDBSequences":            "/api/extract_pdb_sequences",
    "select_pdb_chain":               "/api/select_pdb_chain",
    "SelectPDBChain":                 "/api/select_pdb_chain",
    "calculate_protein_anm":          "/api/calculate_protein_anm",
    "CalculateProteinANM":            "/api/calculate_protein_anm",

    # ---------- 本地Protein FASTA解析 ----------
    "parse_protein_fasta":            "/api/parse_protein_fasta",
    "ParseProteinFASTA":               "/api/parse_protein_fasta",
    "parse_multi_protein_fasta":       "/api/parse_multi_protein_fasta",
    "ParseMultiProteinFASTA":          "/api/parse_multi_protein_fasta",
    "select_fasta_record":             "/api/select_fasta_record",
    "SelectFASTARecord":               "/api/select_fasta_record",
    "align_fasta_records":             "/api/align_fasta_records",
    "AlignFASTARecords":               "/api/align_fasta_records",
    "summarize_pairwise_alignment":    "/api/summarize_pairwise_alignment",
    "SummarizePairwiseAlignment":      "/api/summarize_pairwise_alignment",

    # ---------- 一站式全分析（Agent最常走这条路）----------
    "protein_analysis":               "/api/protein_analysis",
    "ComputeProtPara":                "/api/protein_analysis",
}

PDB_API_PATHS = {
    "/api/analyze_pdb_structure",
    "/api/extract_pdb_sequences",
    "/api/select_pdb_chain",
    "/api/calculate_protein_anm",
}

FASTA_API_PATHS = {
    "/api/parse_protein_fasta",
    "/api/parse_multi_protein_fasta",
}

STRUCTURED_OBJECT_API_PATHS = {
    "/api/select_fasta_record",
    "/api/align_fasta_records",
    "/api/summarize_pairwise_alignment",
}

SEQUENCE_API_PATHS = (
    set(TOOL_NAME_TO_API.values())
    - PDB_API_PATHS
    - FASTA_API_PATHS
    - STRUCTURED_OBJECT_API_PATHS
)

AMINO_ACID_SEQUENCE_TYPE = "amino acid sequence"
PDB_STRUCTURE_FILE_TYPE = "PDB structure file"
PDB_CHAIN_IDENTIFIER_TYPE = "PDB chain identifier"
PROTEIN_FASTA_FILE_TYPE = "single-record protein FASTA file"
MULTI_PROTEIN_FASTA_FILE_TYPE = "multi-record protein FASTA file"
PROTEIN_SEQUENCE_COLLECTION_TYPE = "protein sequence collection"
FASTA_RECORD_IDENTIFIER_TYPE = "FASTA record identifier"
FASTA_RECORD_IDENTIFIER_PAIR_TYPE = "FASTA record identifier pair"
PAIRWISE_PROTEIN_ALIGNMENT_TYPE = "pairwise protein alignment"

# Biology FastAPI 服务地址，统一从 .env 配置读取
BIO_API_BASE = _cfg.BIO_API_BASE

# 单次请求超时（秒），答辩演示时给充裕时间
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)

_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_SEQUENCE_TOKEN_PATTERN = r"[A-Za-z0-9*._-]+"
_UNIPROT_ACCESSION_PATTERN = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])",
    flags=re.IGNORECASE,
)


def _configured_sequence_threshold(name: str, fallback: int) -> int:
    """读取正整数序列定位阈值；配置异常时安全回退。"""
    try:
        return max(1, int(getattr(_cfg, name, fallback)))
    except (TypeError, ValueError):
        return fallback


def _has_identifier_context(text: str, start: int, end: int) -> bool:
    """判断短token是否属于PDB、chain或FASTA record标识语境。"""
    try:
        prefix = text[max(0, start - 40):start]
        suffix = text[end:min(len(text), end + 20)]
        identifier_prefix = re.search(
            r"(?:pdb(?:\s*id)?|chain(?:[\s_-]*id)?|链|"
            r"fasta[\s_-]*record(?:[\s_-]*ids?)?|record(?:[\s_-]*ids?)?|"
            r"记录(?:\s*ID)?)\s*"
            r"(?:is|为|=|:|：)?\s*$",
            prefix,
            flags=re.IGNORECASE,
        )
        chinese_chain_suffix = re.match(r"\s*链(?![A-Za-z0-9])", suffix)
        return bool(identifier_prefix or chinese_chain_suffix)
    except Exception as exc:
        logger.error("判断生物标识符上下文失败: %s", exc, exc_info=True)
        return True


def _looks_like_non_sequence_identifier(token: str) -> bool:
    """识别不应进入氨基酸内容校验的常见生物标识符或文件名。"""
    try:
        normalized = str(token or "").strip().strip("`'\"“”‘’")
        if not normalized:
            return False
        upper = normalized.upper()
        if _UNIPROT_ACCESSION_PATTERN.fullmatch(upper):
            return True
        if upper in {"UNIPROTKB", "SWISS-PROT", "SWISSPROT"}:
            return True
        return bool(
            re.search(
                r"\.(?:FASTA|FA|FAA|PDB|ENT)$",
                upper,
                flags=re.IGNORECASE,
            )
        )
    except Exception as exc:
        logger.error("识别非序列生物标识符失败: %s", exc, exc_info=True)
        return True


def _extract_explicit_sequence_candidate(text: str) -> str | None:
    """定位具有明确序列语义的短token，不在此函数判断氨基酸是否合法。"""
    try:
        minimum_length = _configured_sequence_threshold(
            "EXPLICIT_SEQUENCE_CANDIDATE_MIN_LENGTH",
            1,
        )
        label_pattern = re.compile(
            r"(?:(?:氨基酸|蛋白质)?序列|"
            r"(?:amino\s+acid\s+)?sequences?\b)",
            flags=re.IGNORECASE,
        )
        token_pattern = re.compile(_SEQUENCE_TOKEN_PATTERN)

        # 结构化代码块是最强数据边界，必须先于题面中的“真实序列 P69905”
        # 等描述性标识符。块内允许大小写和空白，定位后统一压缩空白；
        # X/Z/数字等内容仍原样保留给后续标准字符校验。
        block_pattern = re.compile(
            r"```(?:(?:sequence|protein|amino[_ -]?acid)[ \t]*)?\r?\n"
            r"\s*([A-Za-z0-9*._\-\s]+?)\s*\r?\n```",
            flags=re.IGNORECASE,
        )
        for match in block_pattern.finditer(text):
            token = "".join(match.group(1).split())
            if (
                len(token) >= minimum_length
                and not _has_identifier_context(text, match.start(), match.end())
            ):
                return token.upper()

        # 行内代码和引号也是明确数据边界；短候选要求大写，避免把普通
        # 引用英文误当序列。PDB/chain/UniProt标识语境优先排除。
        delimited_patterns = (
            re.compile(rf"`({_SEQUENCE_TOKEN_PATTERN})`"),
            re.compile(rf"[\"'“‘]({_SEQUENCE_TOKEN_PATTERN})[\"'”’]"),
        )
        for pattern in delimited_patterns:
            for match in pattern.finditer(text):
                token = match.group(1)
                if (
                    len(token) >= minimum_length
                    and token == token.upper()
                    and token_pattern.fullmatch(token)
                    and not _looks_like_non_sequence_identifier(token)
                    and not _has_identifier_context(text, match.start(), match.end())
                ):
                    return token.upper()

        # 标签本身提供序列语义，例如“序列 X”或“sequence: X”。
        for label_match in label_pattern.finditer(text):
            remainder = text[label_match.end():]
            strong_match = re.match(
                rf"\s*(?:为|is\b|=|:|：)\s*[`'\"“‘]?"
                rf"({_SEQUENCE_TOKEN_PATTERN})",
                remainder,
                flags=re.IGNORECASE,
            )
            if strong_match:
                token = strong_match.group(1)
                if (
                    len(token) >= minimum_length
                    and not _looks_like_non_sequence_identifier(token)
                ):
                    return token.upper()
            # 兼容“序列的pI：MKW...”这类标签与数据之间带简短字段说明的
            # 明确语义；必须存在冒号边界，避免从普通英文叙述中猜测单词。
            scoped_match = re.match(
                rf"[^\r\n:：]{{0,40}}[:：]\s*[`'\"“‘]?({_SEQUENCE_TOKEN_PATTERN})",
                remainder,
            )
            if scoped_match:
                token = scoped_match.group(1)
                if (
                    len(token) >= minimum_length
                    and not _looks_like_non_sequence_identifier(token)
                ):
                    return token.upper()
            weak_match = re.match(
                rf"\s+[`'\"“‘]?({_SEQUENCE_TOKEN_PATTERN})",
                remainder,
            )
            if weak_match:
                token = weak_match.group(1)
                if (
                    len(token) >= minimum_length
                    and token == token.upper()
                    and not _looks_like_non_sequence_identifier(token)
                ):
                    return token.upper()
        return None
    except Exception as exc:
        logger.error("定位显式短序列候选失败: %s", exc, exc_info=True)
        return None


# ------------------------------------------------------------------ #
#  2. 从自然语言 query 里提取氨基酸序列
#     B 的接口只接受纯序列，不能把整句话传进去
#     规则：连续 10 个以上标准单字母氨基酸字符视为序列
# ------------------------------------------------------------------ #
def extract_sequence_candidate(text: str) -> str | None:
    """严格定位第一段疑似氨基酸序列；找不到时返回None。

    合法序列会标准化为大写；疑似序列中的数字、X/Z等非法字符必须完整
    保留，让Biology服务统一校验。该函数不把普通自然语言回退成序列。
    """
    try:
        if not isinstance(text, str):
            return None
        stripped_text = text.strip()
        if not stripped_text:
            return None

        explicit_candidate = _extract_explicit_sequence_candidate(stripped_text)
        if explicit_candidate is not None:
            return explicit_candidate

        minimum_length = _configured_sequence_threshold(
            "SEQUENCE_CANDIDATE_MIN_LENGTH",
            20,
        )
        minimum_valid_residues = _configured_sequence_threshold(
            "SEQUENCE_CANDIDATE_MIN_VALID_RESIDUES",
            8,
        )
        try:
            minimum_valid_ratio = min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            _cfg,
                            "SEQUENCE_CANDIDATE_MIN_VALID_RATIO",
                            0.6,
                        )
                    ),
                ),
            )
        except (TypeError, ValueError):
            minimum_valid_ratio = 0.6
        raw_matches = re.finditer(r"[A-Za-z0-9*._-]+", stripped_text)

        for raw_match in raw_matches:
            raw_token = raw_match.group(0)
            if raw_token.lower() in {
                "record_id",
                "record_ids",
                "fasta_record_id",
                "fasta_record_ids",
            }:
                continue
            if _has_identifier_context(
                stripped_text,
                raw_match.start(),
                raw_match.end(),
            ):
                continue
            # 英文问题常在序列后直接接句号；句末标点不属于序列内容。
            # 仅移除两端句号，内部句号仍保留并由后端判为非法字符。
            raw_token = raw_token.strip(".")
            if _looks_like_non_sequence_identifier(raw_token):
                continue
            token = raw_token.upper()
            if len(token) < minimum_length:
                continue

            invalid_chars = set(token) - _VALID_AMINO_ACIDS
            if not invalid_chars:
                return token

            valid_count = sum(char in _VALID_AMINO_ACIDS for char in token)
            has_explicit_invalid_marker = any(
                char.isdigit() or char in "*._-" for char in token
            )
            # 数字/分隔符是本次问题的关键：必须把整个候选token传给后端，
            # 不能让后续正则只取到前面的合法氨基酸片段。
            # 纯字母非法候选只由上面的显式语义规则定位；否则大写英文单词
            # 与蛋白序列在字符层不可区分，贸然接收会制造假拒绝。
            if (
                valid_count >= minimum_valid_residues
                and valid_count / len(token) >= minimum_valid_ratio
                and has_explicit_invalid_marker
                and raw_token == raw_token.upper()
            ):
                return token

        return None
    except Exception as exc:
        logger.error("严格提取氨基酸序列候选失败: %s", exc, exc_info=True)
        return None


def extract_sequence(text: str) -> str:
    """
    从用户输入文本中定位第一段疑似氨基酸序列。

    兼容旧baseline：找到候选时返回候选；未找到时仍原样返回text，
    由下游接口生成标准错误。v0.3b不使用该自然语言回退。
    """
    try:
        if not isinstance(text, str):
            return ""
        stripped_text = text.strip()
        if not stripped_text:
            return ""
        candidate = extract_sequence_candidate(stripped_text)
        return candidate if candidate is not None else stripped_text
    except Exception as exc:
        logger.error("提取氨基酸序列失败: %s", exc, exc_info=True)
        return ""


def select_pdb_file(file_path_list: list) -> str | None:
    """从上传文件列表中选择第一份存在的PDB文件。"""
    try:
        if not isinstance(file_path_list, list):
            return None
        for file_path in file_path_list:
            if not isinstance(file_path, str) or not file_path.strip():
                continue
            path = Path(file_path.strip()).expanduser()
            if path.suffix.lower() in {".pdb", ".ent"} and path.is_file():
                return str(path.resolve())
        return None
    except Exception as exc:
        logger.error("选择PDB上传文件失败: %s", exc, exc_info=True)
        return None


def select_fasta_file(file_path_list: list) -> str | None:
    """从上传文件列表中选择第一份存在的Protein FASTA文件。"""
    try:
        if not isinstance(file_path_list, list):
            return None
        for file_path in file_path_list:
            if not isinstance(file_path, str) or not file_path.strip():
                continue
            path = Path(file_path.strip()).expanduser()
            if path.suffix.lower() in {".fasta", ".fa", ".faa"} and path.is_file():
                return str(path.resolve())
        return None
    except Exception as exc:
        logger.error("选择FASTA上传文件失败: %s", exc, exc_info=True)
        return None


# ------------------------------------------------------------------ #
#  3. 工具名解析：KG名 → 接口路径
#     先精确匹配，再做大小写/下划线的模糊兜底
# ------------------------------------------------------------------ #
def resolve_api_path(tool_name: str) -> str | None:
    """
    返回工具对应的接口路径，找不到返回 None。
    优先精确匹配，其次做去格式化后的模糊匹配。
    """
    try:
        if not isinstance(tool_name, str) or not tool_name.strip():
            return None
        # 精确匹配
        if tool_name in TOOL_NAME_TO_API:
            return TOOL_NAME_TO_API[tool_name]

        # 兼容历史大小写和格式分隔符差异，但不猜测不存在的工具功能。
        normalized = re.sub(r"[^a-z0-9]", "", tool_name.lower())
        for key, path in TOOL_NAME_TO_API.items():
            key_normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if key_normalized == normalized:
                return path
        return None
    except Exception as exc:
        logger.error("解析工具接口失败: %s", exc, exc_info=True)
        return None


def find_unknown_tool_names(tool_names: object) -> list[str]:
    """返回计划中无法映射到本地注册接口的去重工具名。"""
    try:
        if not isinstance(tool_names, (list, tuple)):
            return []
        unknown: list[str] = []
        for value in tool_names:
            if not isinstance(value, str) or not value.strip():
                continue
            normalized = value.strip()
            if (
                resolve_api_path(normalized) is None
                and normalized not in unknown
            ):
                unknown.append(normalized)
        return unknown
    except Exception as exc:
        logger.error("识别闭集外工具名失败: %s", exc, exc_info=True)
        return []


def _validate_typed_file(
    value: object,
    allowed_suffixes: set[str],
    label: str,
) -> dict:
    """校验对象池传入的文件对象，并返回受控绝对路径。"""
    try:
        if not isinstance(value, str) or not value.strip():
            return {
                "status": "error",
                "message": f"{label}对象必须是非空文件路径",
            }
        path = Path(value.strip()).expanduser()
        if path.suffix.lower() not in allowed_suffixes:
            return {
                "status": "error",
                "message": f"{label}文件扩展名不受支持",
            }
        if not path.is_file():
            return {
                "status": "error",
                "message": f"{label}文件不存在或不是普通文件",
            }
        return {
            "status": "success",
            "file_path": str(path.resolve()),
        }
    except Exception as exc:
        logger.error("校验类型化文件输入失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"校验{label}文件失败: {exc}",
        }


def _validate_typed_object(
    value: object,
    label: str,
    *,
    schema_version: str | None = None,
) -> dict:
    """校验Artifact Pool传入的结构化对象及可选schema版本。"""
    try:
        if not isinstance(value, dict) or not value:
            return {
                "status": "error",
                "code": "INVALID_TYPED_OBJECT",
                "message": f"{label}对象必须是非空字典",
            }
        if schema_version and value.get("schema_version") != schema_version:
            return {
                "status": "error",
                "code": "UNSUPPORTED_TYPED_OBJECT_SCHEMA",
                "message": f"{label}对象版本不受支持",
            }
        return {"status": "success", "value": value}
    except Exception as exc:
        logger.error("校验类型化结构对象失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "code": "TYPED_OBJECT_VALIDATION_EXCEPTION",
            "message": f"校验{label}对象失败: {exc}",
        }
def build_typed_tool_payload(
    api_path: str,
    typed_inputs: object,
    tool_parameters: object = None,
) -> dict:
    """把Artifact Pool真实对象转换为Biology接口请求体。

    本函数只做类型到参数的确定性映射，不从自然语言再次提取数据。
    """
    try:
        if not isinstance(api_path, str) or not api_path.strip():
            return {"status": "error", "message": "接口路径不能为空"}
        if not isinstance(typed_inputs, dict) or not typed_inputs:
            return {
                "status": "error",
                "message": "typed_inputs必须是非空类型到真实值映射",
            }

        parameter_result = validate_tool_parameters(api_path, tool_parameters or {})
        if parameter_result.get("status") != "success":
            return parameter_result
        explicit_parameters = parameter_result.get("parameters") or {}

        if api_path in PDB_API_PATHS:
            file_result = _validate_typed_file(
                typed_inputs.get(PDB_STRUCTURE_FILE_TYPE),
                {".pdb", ".ent"},
                "PDB",
            )
            if file_result.get("status") != "success":
                return file_result
            json_data = {"file_path": file_result["file_path"]}
            if api_path == "/api/select_pdb_chain":
                chain_id = typed_inputs.get(PDB_CHAIN_IDENTIFIER_TYPE)
                if not isinstance(chain_id, str) or not chain_id.strip():
                    return {
                        "status": "error",
                        "message": "PDB单链选择缺少类型化chain_id对象",
                    }
                explicit_chain = explicit_parameters.get("chain_id")
                if explicit_chain and explicit_chain != chain_id.strip():
                    return {
                        "status": "error",
                        "code": "TOOL_PARAMETER_ARTIFACT_CONFLICT",
                        "message": "显式chain_id与类型化链对象冲突",
                    }
                json_data["chain_id"] = chain_id.strip()
            elif api_path == "/api/calculate_protein_anm":
                json_data.update(explicit_parameters)
            return {
                "status": "success",
                "json_data": json_data,
                "input_types": sorted(str(key) for key in typed_inputs),
            }

        if api_path in FASTA_API_PATHS:
            fasta_type = (
                MULTI_PROTEIN_FASTA_FILE_TYPE
                if api_path == "/api/parse_multi_protein_fasta"
                else PROTEIN_FASTA_FILE_TYPE
            )
            file_result = _validate_typed_file(
                typed_inputs.get(fasta_type),
                {".fasta", ".fa", ".faa"},
                "Protein FASTA",
            )
            if file_result.get("status") != "success":
                return file_result
            return {
                "status": "success",
                "json_data": {"file_path": file_result["file_path"]},
                "input_types": sorted(str(key) for key in typed_inputs),
            }

        if api_path == "/api/select_fasta_record":
            collection_result = _validate_typed_object(
                typed_inputs.get(PROTEIN_SEQUENCE_COLLECTION_TYPE),
                "ProteinSequenceCollection",
                schema_version="protein-sequence-collection/v1",
            )
            if collection_result.get("status") != "success":
                return collection_result
            record_id = typed_inputs.get(FASTA_RECORD_IDENTIFIER_TYPE)
            if not isinstance(record_id, str) or not record_id.strip():
                return {
                    "status": "error",
                    "code": "MISSING_FASTA_RECORD_IDENTIFIER",
                    "message": "FASTA记录选择缺少类型化record_id对象",
                }
            return {
                "status": "success",
                "json_data": {
                    "collection": collection_result["value"],
                    "record_id": record_id.strip(),
                },
                "input_types": sorted(str(key) for key in typed_inputs),
            }

        if api_path == "/api/align_fasta_records":
            collection_result = _validate_typed_object(
                typed_inputs.get(PROTEIN_SEQUENCE_COLLECTION_TYPE),
                "ProteinSequenceCollection",
                schema_version="protein-sequence-collection/v1",
            )
            if collection_result.get("status") != "success":
                return collection_result
            record_ids = typed_inputs.get(FASTA_RECORD_IDENTIFIER_PAIR_TYPE)
            if (
                not isinstance(record_ids, (list, tuple))
                or len(record_ids) != 2
                or not all(isinstance(item, str) and item.strip() for item in record_ids)
                or record_ids[0].strip() == record_ids[1].strip()
            ):
                return {
                    "status": "error",
                    "code": "INVALID_FASTA_RECORD_IDENTIFIER_PAIR",
                    "message": "FASTA比对缺少两个不同的类型化record_id",
                }
            return {
                "status": "success",
                "json_data": {
                    "collection": collection_result["value"],
                    "record_ids": [item.strip() for item in record_ids],
                },
                "input_types": sorted(str(key) for key in typed_inputs),
            }

        if api_path == "/api/summarize_pairwise_alignment":
            alignment_result = _validate_typed_object(
                typed_inputs.get(PAIRWISE_PROTEIN_ALIGNMENT_TYPE),
                "PairwiseProteinAlignment",
                schema_version="pairwise-protein-alignment/v1",
            )
            if alignment_result.get("status") != "success":
                return alignment_result
            return {
                "status": "success",
                "json_data": {"alignment": alignment_result["value"]},
                "input_types": sorted(str(key) for key in typed_inputs),
            }

        if api_path not in SEQUENCE_API_PATHS:
            return {
                "status": "error",
                "code": "UNREGISTERED_TYPED_API",
                "message": "接口不在已注册的类型化输入映射中",
            }

        sequence = typed_inputs.get(AMINO_ACID_SEQUENCE_TYPE)
        if not isinstance(sequence, str) or not sequence.strip():
            return {
                "status": "error",
                "message": "序列工具缺少类型化氨基酸序列对象",
            }
        return {
            "status": "success",
            "json_data": {"sequence": sequence.strip()},
            "input_types": sorted(str(key) for key in typed_inputs),
        }
    except Exception as exc:
        logger.error("构造类型化工具请求失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": f"构造类型化工具请求失败: {exc}",
        }


def _parse_json_object(value: object) -> dict:
    """从Runner输入或前序结果中保守解析一个JSON对象，不执行任意代码。"""
    try:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value.strip():
            return {}
        text = value.strip()
        candidates = [text]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            candidates.append(fenced.group(1))
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidates.append(text[first_brace:last_brace + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return {}
    except Exception as exc:
        logger.error("解析Runner结构化JSON失败: %s", exc, exc_info=True)
        return {}


def _build_untyped_structured_payload(
    api_path: str,
    tool_args: str,
    previous_tool_output: object,
) -> dict:
    """为baseline保守绑定前序结构对象和用户显式记录ID，不做类型修复。"""
    try:
        args_object = _parse_json_object(tool_args)
        previous_object = _parse_json_object(previous_tool_output)
        source = previous_object or args_object
        identifier_result = extract_fasta_record_identifiers(tool_args)

        if api_path == "/api/select_fasta_record":
            collection = source.get("collection")
            record_id = args_object.get("record_id")
            if not isinstance(record_id, str):
                ids = identifier_result.get("record_ids") or []
                record_id = ids[0] if identifier_result.get("mode") == "single" else None
            if not isinstance(collection, dict) or not isinstance(record_id, str):
                return {
                    "status": "error",
                    "code": "MISSING_STRUCTURED_TOOL_INPUT",
                    "message": "select_fasta_record需要前序collection和显式record_id",
                }
            return {
                "status": "success",
                "json_data": {"collection": collection, "record_id": record_id},
            }

        if api_path == "/api/align_fasta_records":
            collection = source.get("collection")
            record_ids = args_object.get("record_ids")
            if not isinstance(record_ids, list):
                record_ids = identifier_result.get("record_ids")
            if not isinstance(collection, dict) or not isinstance(record_ids, list):
                return {
                    "status": "error",
                    "code": "MISSING_STRUCTURED_TOOL_INPUT",
                    "message": "align_fasta_records需要前序collection和两个显式record_id",
                }
            return {
                "status": "success",
                "json_data": {"collection": collection, "record_ids": record_ids},
            }

        if api_path == "/api/summarize_pairwise_alignment":
            alignment = source.get("alignment")
            if not isinstance(alignment, dict):
                return {
                    "status": "error",
                    "code": "MISSING_STRUCTURED_TOOL_INPUT",
                    "message": "summarize_pairwise_alignment需要前序alignment对象",
                }
            return {"status": "success", "json_data": {"alignment": alignment}}

        return {
            "status": "error",
            "code": "UNREGISTERED_STRUCTURED_API",
            "message": "结构化接口未注册baseline输入绑定",
        }
    except Exception as exc:
        logger.error("构造baseline结构化请求失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "code": "STRUCTURED_PAYLOAD_EXCEPTION",
            "message": f"构造baseline结构化请求失败: {exc}",
        }
# ------------------------------------------------------------------ #
#  4. 主函数：异步调用工具
# ------------------------------------------------------------------ #
async def run_tool(
    tool_name: str,
    tool_args: str,
    file_path_list: list,
    typed_inputs: dict | None = None,
    previous_tool_output: object = None,
    tool_parameters: dict | None = None,
) -> str:
    """
    将 Agent 规划出的工具调用转发给 Biology FastAPI 服务。

    Args:
        tool_name:      KG 检索返回的工具名
        tool_args:      用户原始问题或上一步工具输出（会自动提取序列）
        file_path_list: 上传文件列表（文件工具按扩展名选择PDB或FASTA）
        typed_inputs:   可选的类型到真实值映射；提供后禁止再次从文本猜测输入
        previous_tool_output: baseline可选的前序工具JSON；仅供结构对象确定性传递
        tool_parameters: 可选的请求级显式参数上下文；普通与类型化路径共用

    Returns:
        str: 工具执行结果（成功时为 JSON 字符串，失败时为 error 描述）
    """
    try:
        # --- Step 1: 解析接口路径；参数解析也必须处于异常边界内。---
        api_path = resolve_api_path(tool_name)
        if api_path is None:
            msg = (f"工具 '{tool_name}' 未在 Biology 模块接口映射表中，"
                   f"请联系队员D更新 tool_name_dict.py 和 tool_runner.py。")
            logger.warning(f"[tool_runner] {msg}")
            return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)

        url = f"{BIO_API_BASE}{api_path}"

        # --- Step 2: 根据工具输入类型构造请求。---
        normalized_args = tool_args if isinstance(tool_args, str) else str(tool_args or "")
        parameter_context = (
            tool_parameters
            if tool_parameters is not None
            else build_tool_parameter_context(normalized_args)
        )
        parameter_result = resolve_tool_parameters(parameter_context, api_path)
        if parameter_result.get("status") != "success":
            msg = parameter_result.get("message", "工具参数无效")
            issue = (parameter_result.get("issues") or [{}])[0]
            logger.warning("[tool_runner] %s", msg)
            return json.dumps(
                {
                    "status": "error",
                    "code": issue.get("code", "INVALID_TOOL_PARAMETER"),
                    "message": msg,
                    "issues": parameter_result.get("issues") or [],
                },
                ensure_ascii=False,
            )
        explicit_parameters = parameter_result.get("parameters") or {}
        if typed_inputs is not None:
            typed_payload = build_typed_tool_payload(
                api_path,
                typed_inputs,
                explicit_parameters,
            )
            if typed_payload.get("status") != "success":
                msg = typed_payload.get("message", "类型化工具输入无效")
                logger.warning("[tool_runner] %s", msg)
                error_payload = {
                    "status": "error",
                    "code": typed_payload.get("code", "INVALID_TYPED_INPUT"),
                    "message": msg,
                }
                if "issues" in typed_payload:
                    error_payload["issues"] = typed_payload["issues"]
                return json.dumps(
                    error_payload,
                    ensure_ascii=False,
                )
            json_data = typed_payload["json_data"]
            logger.info(
                "[tool_runner] 使用Artifact Pool输入 | 接口: %s | types=%s",
                url,
                typed_payload.get("input_types"),
            )
        elif api_path in PDB_API_PATHS:
            pdb_file = select_pdb_file(file_path_list)
            if pdb_file is None:
                msg = "PDB工具需要先在前端上传有效的.pdb文件"
                logger.warning("[tool_runner] %s", msg)
                return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)
            json_data = {"file_path": pdb_file}
            if api_path == "/api/calculate_protein_anm":
                json_data.update(explicit_parameters)
            elif api_path == "/api/select_pdb_chain":
                chain_id = explicit_parameters.get("chain_id")
                if not chain_id:
                    msg = "PDB单链选择需要用户在问题中明确提供chain_id"
                    logger.warning("[tool_runner] %s", msg)
                    return json.dumps(
                        {"status": "error", "message": msg},
                        ensure_ascii=False,
                    )
                json_data["chain_id"] = chain_id
            logger.info(
                "[tool_runner] 调用接口: %s | PDB文件: %s",
                url,
                Path(pdb_file).name,
            )
        elif api_path in FASTA_API_PATHS:
            fasta_file = select_fasta_file(file_path_list)
            if fasta_file is None:
                msg = "FASTA工具需要先在前端上传有效的.fasta/.fa/.faa文件"
                logger.warning("[tool_runner] %s", msg)
                return json.dumps(
                    {"status": "error", "message": msg},
                    ensure_ascii=False,
                )
            json_data = {"file_path": fasta_file}
            logger.info(
                "[tool_runner] 调用接口: %s | FASTA文件: %s",
                url,
                Path(fasta_file).name,
            )
        elif api_path in STRUCTURED_OBJECT_API_PATHS:
            structured_payload = _build_untyped_structured_payload(
                api_path,
                normalized_args,
                previous_tool_output,
            )
            if structured_payload.get("status") != "success":
                return json.dumps(structured_payload, ensure_ascii=False)
            json_data = structured_payload["json_data"]
            logger.info(
                "[tool_runner] 调用结构化接口: %s | fields=%s",
                url,
                sorted(json_data),
            )
        else:
            sequence = extract_sequence(normalized_args)
            if not sequence:
                msg = "工具输入中未找到非空氨基酸序列"
                logger.warning("[tool_runner] %s", msg)
                return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)
            json_data = {"sequence": sequence}
            logger.info(
                "[tool_runner] 调用接口: %s | 序列长度: %s | 原始输入长度: %s",
                url,
                len(sequence),
                len(normalized_args),
            )

        # --- Step 3: 发起请求，全面捕获异常。---
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(url, json=json_data) as response:
                if response.status == 200:
                    res = await response.json()
                    logger.info(f"[tool_runner] 调用成功: {tool_name}")
                    return json.dumps(res, ensure_ascii=False)
                else:
                    # HTTP 错误（4xx/5xx），读取错误体返回，不抛异常
                    err_text = await response.text()
                    msg = (f"Biology服务返回HTTP {response.status}，"
                           f"接口: {api_path}，详情: {err_text[:200]}")
                    logger.error(f"[tool_runner] {msg}")
                    return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)

    except aiohttp.ClientConnectorError:
        msg = (f"无法连接 Biology 服务 ({BIO_API_BASE})，"
               f"请确认 main.py 已在 :8000 端口启动。")
        logger.error(f"[tool_runner] 连接失败: {msg}")
        return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)

    except aiohttp.ServerTimeoutError:
        msg = f"Biology 服务请求超时（>{REQUEST_TIMEOUT.total}s），接口: {api_path}"
        logger.error(f"[tool_runner] 超时: {msg}")
        return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)

    except Exception as e:
        msg = f"tool_runner 未预期异常 [{type(e).__name__}]: {str(e)}"
        logger.error(f"[tool_runner] {msg}", exc_info=True)
        return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)
