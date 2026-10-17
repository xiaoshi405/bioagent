# -*- coding: utf-8 -*-
"""Build tool payloads and call the protein API."""

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

TOOL_NAME_TO_API: dict[str, str] = {
    "ComputePiMw":                    "/api/compute_pi_mw",
    "compute_pI_mw":                  "/api/compute_pi_mw",
    "ProteinIsoelectricPointCalculator": "/api/compute_pi_mw",

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

    "PredictSignalPeptide":           "/api/predict_signalpeptide",
    "predict_signalpeptide":          "/api/predict_signalpeptide",

    "PredictTransmembraneRegion":     "/api/predict_transmembrane",
    "predict_transmembrane":          "/api/predict_transmembrane",

    "ProteinSolubilityPredictor":     "/api/predict_solubility",
    "PredictProteinSolubility":       "/api/predict_solubility",
    "predict_solubility":             "/api/predict_solubility",

    "check_protein_toxicity":         "/api/check_protein_toxicity",
    "CheckProteinToxicity":           "/api/check_protein_toxicity",

    "analyze_pdb_structure":          "/api/analyze_pdb_structure",
    "AnalyzeProteinStructure":        "/api/analyze_pdb_structure",
    "extract_pdb_sequences":          "/api/extract_pdb_sequences",
    "ExtractPDBSequences":            "/api/extract_pdb_sequences",
    "select_pdb_chain":               "/api/select_pdb_chain",
    "SelectPDBChain":                 "/api/select_pdb_chain",
    "calculate_protein_anm":          "/api/calculate_protein_anm",
    "CalculateProteinANM":            "/api/calculate_protein_anm",

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

BIO_API_BASE = _cfg.BIO_API_BASE

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60)

_VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_SEQUENCE_TOKEN_PATTERN = r"[A-Za-z0-9*._-]+"
_UNIPROT_ACCESSION_PATTERN = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])",
    flags=re.IGNORECASE,
)


def _configured_sequence_threshold(name: str, fallback: int) -> int:
    try:
        return max(1, int(getattr(_cfg, name, fallback)))
    except (TypeError, ValueError):
        return fallback


def _has_identifier_context(text: str, start: int, end: int) -> bool:
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

        # Prefer explicit code blocks over descriptive identifiers.
        # Keep invalid characters for downstream validation.
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

        # Quoted short candidates must be uppercase; exclude identifier contexts.
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

        # A sequence label provides an explicit sequence boundary.
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
            # Allow a short field label before a colon-delimited sequence.
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

def extract_sequence_candidate(text: str) -> str | None:
    """Locate a sequence candidate and preserve invalid characters for validation."""
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
            # Strip sentence periods at the ends, not inside the sequence.
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
            # Keep the whole invalid candidate instead of extracting a valid prefix.
            # Require explicit context for invalid alphabetic candidates.
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
    """Locate a sequence; retain the legacy text fallback if none is found."""
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

def resolve_api_path(tool_name: str) -> str | None:
    try:
        if not isinstance(tool_name, str) or not tool_name.strip():
            return None
        if tool_name in TOOL_NAME_TO_API:
            return TOOL_NAME_TO_API[tool_name]

        # Allow legacy spelling separators, but do not infer unknown tools.
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
    """Map artifact values to API parameters without parsing natural language again."""
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

async def run_tool(
    tool_name: str,
    tool_args: str,
    file_path_list: list,
    typed_inputs: dict | None = None,
    previous_tool_output: object = None,
    tool_parameters: dict | None = None,
) -> str:
    """Build a request and call the registered protein tool endpoint."""
    try:
        api_path = resolve_api_path(tool_name)
        if api_path is None:
            msg = (f"工具 '{tool_name}' 未在 Biology 模块接口映射表中，"
                   f"请联系队员D更新 tool_name_dict.py 和 tool_runner.py。")
            logger.warning(f"[tool_runner] {msg}")
            return json.dumps({"status": "error", "message": msg}, ensure_ascii=False)

        url = f"{BIO_API_BASE}{api_path}"

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

        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(url, json=json_data) as response:
                if response.status == 200:
                    res = await response.json()
                    logger.info(f"[tool_runner] 调用成功: {tool_name}")
                    return json.dumps(res, ensure_ascii=False)
                else:
                    # Return the error body for HTTP failures.
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
