# -*- coding: utf-8 -*-
"""Protein tool API."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import traceback

root_dir = Path(__file__).resolve().parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app.core.config import Config
from app.core.logger import get_logger
from app.core.runtime_identity import (
    capture_runtime_identity,
    read_runtime_identity_snapshot,
)

try:
    from . import biologyCommon as bio
    from . import fasta_analysis as fasta_tools
    from . import fasta_collection as fasta_collection_tools
    from . import pdb_analysis as pdb_tools
    from . import protein_alignment as alignment_tools
except ImportError:
    import biologyCommon as bio
    import fasta_analysis as fasta_tools
    import fasta_collection as fasta_collection_tools
    import pdb_analysis as pdb_tools
    import protein_alignment as alignment_tools

_cfg = Config()
logger = get_logger("BiologyService", "biology.log")
_SERVICE_STARTED_AT = datetime.now(timezone.utc).isoformat()
_SERVICE_RUNTIME_IDENTITY = capture_runtime_identity(
    service_name="biology",
    project_root=root_dir,
    configured_port=_cfg.BIO_API_PORT,
    started_at=_SERVICE_STARTED_AT,
)

app = FastAPI(title="生命科学模块API接口 - 成员B完成", version="1.0")

class ProteinSequence(BaseModel):
    sequence: str


class PDBFileRequest(BaseModel):

    file_path: str
    n_modes: int = Field(default=10, ge=1, le=20)
    cutoff: float = Field(default=12.0, ge=4.0, le=20.0)
    chain_id: str | None = Field(default=None, max_length=8)


class PDBChainRequest(BaseModel):

    file_path: str
    chain_id: str = Field(min_length=1, max_length=8)


class FASTAFileRequest(BaseModel):

    file_path: str


class FASTARecordSelectionRequest(BaseModel):

    collection: dict[str, Any]
    record_id: str = Field(min_length=1, max_length=256)


class FASTARecordPairRequest(BaseModel):

    collection: dict[str, Any]
    record_ids: list[str] = Field(min_length=2, max_length=2)


class PairwiseAlignmentRequest(BaseModel):

    alignment: dict[str, Any]


def _format_tool_response(result):
    try:
        if not isinstance(result, dict):
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "工具返回格式无效"},
            )
        if result.get("status") == "error":
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as exc:
        logger.error("格式化工具响应失败: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"格式化工具响应失败: {exc}"},
        )


def _execute_sequence_tool(tool_name, sequence, tool_function):
    try:
        logger.info("调用 %s | 序列长度: %s", tool_name, len(sequence))
        result = tool_function(sequence)
        if not isinstance(result, dict):
            result = {"status": "error", "message": "工具返回格式无效"}
        if result.get("status") == "error":
            logger.warning("%s 返回错误: %s", tool_name, result.get("message"))
        else:
            logger.info("%s 执行成功", tool_name)
        return _format_tool_response(result)
    except Exception as exc:
        logger.error("%s 异常: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{tool_name} 执行失败: {exc}"},
        )


def _execute_pdb_tool(tool_name, file_path, tool_function, **kwargs):
    try:
        file_name = Path(str(file_path)).name
        logger.info("调用 %s | PDB文件: %s", tool_name, file_name)
        result = tool_function(file_path, **kwargs)
        if not isinstance(result, dict):
            result = {"status": "error", "message": "工具返回格式无效"}
        if result.get("status") == "error":
            logger.warning("%s 返回错误: %s", tool_name, result.get("message"))
        else:
            logger.info("%s 执行成功", tool_name)
        return _format_tool_response(result)
    except Exception as exc:
        logger.error("%s 异常: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{tool_name} 执行失败: {exc}"},
        )


def _execute_fasta_tool(tool_name, file_path, tool_function):
    try:
        file_name = Path(str(file_path)).name
        logger.info("调用 %s | FASTA文件: %s", tool_name, file_name)
        result = tool_function(file_path)
        if not isinstance(result, dict):
            result = {"status": "error", "message": "工具返回格式无效"}
        if result.get("status") == "error":
            logger.warning("%s 返回错误: %s", tool_name, result.get("message"))
        else:
            logger.info("%s 执行成功", tool_name)
        return _format_tool_response(result)
    except Exception as exc:
        logger.error("%s 异常: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"{tool_name} 执行失败: {exc}"},
        )


def _execute_object_tool(tool_name, tool_function, **kwargs):
    try:
        logger.info("调用 %s | 输入字段: %s", tool_name, sorted(kwargs))
        result = tool_function(**kwargs)
        if not isinstance(result, dict):
            result = {
                "status": "error",
                "code": "INVALID_TOOL_RESULT",
                "message": "工具返回格式无效",
                "issues": [],
            }
        if result.get("status") == "error":
            logger.warning("%s 返回错误: %s", tool_name, result.get("message"))
        else:
            logger.info("%s 执行成功", tool_name)
        return _format_tool_response(result)
    except Exception as exc:
        logger.error("%s 异常: %s", tool_name, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "code": "OBJECT_TOOL_EXECUTION_EXCEPTION",
                "message": f"{tool_name} 执行失败: {exc}",
                "issues": [],
            },
        )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        logger.warning("请求体验证失败 | path=%s | errors=%s", request.url.path, exc.errors())
        is_pdb_endpoint = request.url.path in {
            "/api/analyze_pdb_structure",
            "/api/extract_pdb_sequences",
            "/api/select_pdb_chain",
            "/api/calculate_protein_anm",
        }
        is_fasta_endpoint = request.url.path in {
            "/api/parse_protein_fasta",
            "/api/parse_multi_protein_fasta",
        }
        is_collection_endpoint = request.url.path in {
            "/api/select_fasta_record",
            "/api/align_fasta_records",
        }
        if request.url.path == "/api/select_pdb_chain":
            message = "请求体必须包含字符串 file_path 和 chain_id 字段"
        elif is_pdb_endpoint:
            message = "请求体必须包含字符串 file_path 字段，ANM参数需在允许范围内"
        elif is_fasta_endpoint:
            message = "请求体必须包含字符串 file_path 字段"
        elif request.url.path == "/api/summarize_pairwise_alignment":
            message = "请求体必须包含alignment字典对象"
        elif is_collection_endpoint:
            message = "请求体必须包含collection字典及显式FASTA记录ID字段"
        else:
            message = "请求体必须包含字符串 sequence 字段"
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": message},
        )
    except Exception as handler_exc:
        logger.error("处理请求体验证错误失败: %s", handler_exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"请求体验证处理失败: {handler_exc}"},
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    logger.info(f"收到 API 请求: {method} {path}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"完成 API 请求: {method} {path} | 状态码: {response.status_code} | 耗时: {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"处理 API 请求失败: {method} {path} | 耗时: {process_time:.4f}s | 异常: {type(e).__name__}: {str(e)}", exc_info=True)
        raise e

@app.get("/")
def home():
    return {"message": "生命科学分析API服务运行正常", "author": "成员B"}


@app.get("/api/runtime-info")
def runtime_info():
    try:
        return read_runtime_identity_snapshot(_SERVICE_RUNTIME_IDENTITY)
    except Exception as exc:
        logger.error("读取Biology运行身份失败: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"读取Biology运行身份失败: {exc}"},
        )

@app.post("/api/compute_pi_mw", summary="计算等电点 & 分子量")
def api_compute_pi_mw(data: ProteinSequence):
    logger.info(f"调用 compute_pi_mw | 序列长度: {len(data.sequence)}")
    try:
        result = bio.compute_pI_mw(data.sequence)
        if result.get("status") == "error":
            logger.warning(f"compute_pi_mw 返回错误: {result.get('message')}")
        else:
            logger.info(f"compute_pi_mw 计算成功 | pI: {result.get('pI')} | Mw: {result.get('molecular_weight')}")
        return _format_tool_response(result)
    except Exception as e:
        logger.error(f"compute_pi_mw 异常: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/api/compute_gravy", summary="计算蛋白质GRAVY值")
def api_compute_gravy(data: ProteinSequence):
    """Compute GRAVY on the Kyte-Doolittle scale."""
    try:
        return _execute_sequence_tool("compute_gravy", data.sequence, bio.compute_gravy)
    except Exception as exc:
        logger.error("compute_gravy 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/compute_secondary_structure_fraction", summary="计算二级结构倾向比例")
def api_compute_secondary_structure_fraction(data: ProteinSequence):
    """Estimate helix, turn and sheet propensities from residue composition."""
    try:
        return _execute_sequence_tool(
            "compute_secondary_structure_fraction",
            data.sequence,
            bio.compute_secondary_structure_fraction,
        )
    except Exception as exc:
        logger.error("compute_secondary_structure_fraction 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/compute_amino_acid_composition", summary="计算氨基酸组成")
def api_compute_amino_acid_composition(data: ProteinSequence):
    """Count the 20 standard amino acids and their percentages."""
    try:
        return _execute_sequence_tool(
            "compute_amino_acid_composition",
            data.sequence,
            bio.compute_amino_acid_composition,
        )
    except Exception as exc:
        logger.error("compute_amino_acid_composition 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/compute_instability_index", summary="计算蛋白质不稳定性指数")
def api_compute_instability_index(data: ProteinSequence):
    """Compute the instability index; use 40 as the empirical cutoff."""
    try:
        return _execute_sequence_tool(
            "compute_instability_index", data.sequence, bio.compute_instability_index
        )
    except Exception as exc:
        logger.error("compute_instability_index 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/compute_extinction_coefficient", summary="计算蛋白质消光系数")
def api_compute_extinction_coefficient(data: ProteinSequence):
    """Compute reduced and oxidized molar extinction coefficients at 280 nm."""
    try:
        return _execute_sequence_tool(
            "compute_extinction_coefficient",
            data.sequence,
            bio.compute_extinction_coefficient,
        )
    except Exception as exc:
        logger.error("compute_extinction_coefficient 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/check_protein_toxicity", summary="筛查毒性蛋白高相似序列")
def api_check_protein_toxicity(data: ProteinSequence):
    """Screen against the local toxicity database using Smith-Waterman alignment."""
    try:
        return _execute_sequence_tool(
            "check_protein_toxicity", data.sequence, bio.check_protein_toxicity
        )
    except Exception as exc:
        logger.error("check_protein_toxicity 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/analyze_pdb_structure", summary="分析PDB结构摘要")
def api_analyze_pdb_structure(data: PDBFileRequest):
    """Summarize chains, residues, atoms, ligands and geometry."""
    try:
        return _execute_pdb_tool(
            "analyze_pdb_structure",
            data.file_path,
            pdb_tools.analyze_pdb_structure,
        )
    except Exception as exc:
        logger.error("analyze_pdb_structure 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/parse_protein_fasta", summary="解析单记录Protein FASTA")
def api_parse_protein_fasta(data: FASTAFileRequest):
    """Read exactly one standard protein FASTA record."""
    try:
        return _execute_fasta_tool(
            "parse_protein_fasta",
            data.file_path,
            fasta_tools.parse_protein_fasta,
        )
    except Exception as exc:
        logger.error("parse_protein_fasta 接口异常: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post("/api/parse_multi_protein_fasta", summary="解析多记录Protein FASTA")
def api_parse_multi_protein_fasta(data: FASTAFileRequest):
    """Read at least two FASTA records into a named collection."""
    try:
        return _execute_fasta_tool(
            "parse_multi_protein_fasta",
            data.file_path,
            fasta_collection_tools.parse_multi_protein_fasta,
        )
    except Exception as exc:
        logger.error("parse_multi_protein_fasta 接口异常: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post("/api/select_fasta_record", summary="按record_id选择FASTA记录")
def api_select_fasta_record(data: FASTARecordSelectionRequest):
    """Select the exact requested record ID; do not use approximate matching."""
    return _execute_object_tool(
        "select_fasta_record",
        fasta_collection_tools.select_fasta_record,
        collection=data.collection,
        record_id=data.record_id,
    )


@app.post("/api/align_fasta_records", summary="比对两个FASTA记录")
def api_align_fasta_records(data: FASTARecordPairRequest):
    """Globally align two explicit records using the configured BLOSUM matrix."""
    return _execute_object_tool(
        "align_fasta_records",
        alignment_tools.align_fasta_records,
        collection=data.collection,
        record_ids=data.record_ids,
    )


@app.post("/api/summarize_pairwise_alignment", summary="汇总蛋白质比对统计")
def api_summarize_pairwise_alignment(data: PairwiseAlignmentRequest):
    """Summarize identity, similarity, gaps and coverage."""
    return _execute_object_tool(
        "summarize_pairwise_alignment",
        alignment_tools.summarize_pairwise_alignment,
        alignment=data.alignment,
    )


@app.post("/api/extract_pdb_sequences", summary="提取PDB蛋白质链序列")
def api_extract_pdb_sequences(data: PDBFileRequest):
    """Extract chain sequences from the first PDB model."""
    try:
        return _execute_pdb_tool(
            "extract_pdb_sequences",
            data.file_path,
            pdb_tools.extract_pdb_sequences,
        )
    except Exception as exc:
        logger.error("extract_pdb_sequences 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/select_pdb_chain", summary="按链ID选择PDB蛋白质序列")
def api_select_pdb_chain(data: PDBChainRequest):
    """Return the sequence of the explicitly requested chain."""
    try:
        return _execute_pdb_tool(
            "select_pdb_chain",
            data.file_path,
            pdb_tools.select_pdb_chain,
            chain_id=data.chain_id,
        )
    except Exception as exc:
        logger.error("select_pdb_chain 接口异常: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@app.post("/api/calculate_protein_anm", summary="计算蛋白质Cα各向异性网络模型")
def api_calculate_protein_anm(data: PDBFileRequest):
    """Compute the lowest nonzero ANM modes from one chain of C-alpha atoms."""
    try:
        return _execute_pdb_tool(
            "calculate_protein_anm",
            data.file_path,
            pdb_tools.calculate_protein_anm,
            n_modes=data.n_modes,
            cutoff=data.cutoff,
            chain_id=data.chain_id,
        )
    except Exception as exc:
        logger.error("calculate_protein_anm 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})

@app.post("/api/predict_signalpeptide", summary="信号肽预测")
def api_predict_signalpeptide(data: ProteinSequence):
    logger.info(f"调用 predict_signalpeptide | 序列长度: {len(data.sequence)}")
    try:
        result = bio.predict_signalpeptide(data.sequence)
        if result.get("status") == "error":
            logger.warning(f"predict_signalpeptide 返回错误: {result.get('message')}")
        else:
            logger.info(f"predict_signalpeptide 预测完成 | has_signal_peptide: {result.get('has_signal_peptide')}")
        return _format_tool_response(result)
    except Exception as e:
        logger.error(f"predict_signalpeptide 异常: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/predict_transmembrane", summary="跨膜区预测")
def api_predict_transmembrane(data: ProteinSequence):
    logger.info(f"调用 predict_transmembrane | 序列长度: {len(data.sequence)}")
    try:
        result = bio.predict_transmembrane(data.sequence)
        if result.get("status") == "error":
            logger.warning(f"predict_transmembrane 返回错误: {result.get('message')}")
        else:
            logger.info(f"predict_transmembrane 预测完成 | 跨膜螺旋数: {result.get('count')}")
        return _format_tool_response(result)
    except Exception as e:
        logger.error(f"predict_transmembrane 异常: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/predict_solubility", summary="蛋白溶解度预测")
def api_predict_solubility(data: ProteinSequence):
    logger.info(f"调用 predict_solubility | 序列长度: {len(data.sequence)}")
    try:
        result = bio.predict_solubility(data.sequence)
        if result.get("status") == "error":
            logger.warning(f"predict_solubility 返回错误: {result.get('message')}")
        else:
            logger.info(f"predict_solubility 预测完成 | soluble: {result.get('soluble')} | score: {result.get('solubility_score')}")
        return _format_tool_response(result)
    except Exception as e:
        logger.error(f"predict_solubility 异常: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/protein_analysis", summary="一站式蛋白全分析（Agent专用）")
def api_protein_analysis(data: ProteinSequence):
    logger.info(f"调用一站式 protein_analysis | 序列长度: {len(data.sequence)}")
    try:
        result = bio.protein_analysis(data.sequence)
        if result.get("status") == "error":
            logger.warning(f"protein_analysis 返回错误: {result.get('message')}")
        else:
            logger.info("protein_analysis 全分析完成")
        return _format_tool_response(result)
    except Exception as e:
        logger.error(f"protein_analysis 异常: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    logger.info(f"正在启动生命科学 API 服务，监听端口: {_cfg.BIO_API_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=_cfg.BIO_API_PORT)
