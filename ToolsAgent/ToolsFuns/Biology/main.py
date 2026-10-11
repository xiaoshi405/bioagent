# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 16:10:21 2026

@author: huawei

修改记录：
- 2026-07-17：兼容包导入与脚本启动；统一工具错误为标准 JSON 响应，
  避免 HTTP 400 被通用异常处理错误转换成 HTTP 500。
- 2026-07-19：接入五个 Biopython 序列描述工具和本地毒性相似性筛查接口；
  统一请求体验证错误的 JSON 结构。
- 2026-07-20：接入 PDB 结构摘要、序列提取和 Cα-ANM 三个离线文件工具。
- 2026-07-24：接入单记录 Protein FASTA 离线解析接口。
- 2026-07-24：增加需要显式chain_id的PDB单链选择接口，禁止多链自动猜测。
- 2026-08-11：新增只读运行身份端点，供正式实验核对实际运行工作树。
- 2026-08-14：接入多记录FASTA集合、显式记录选择、成对比对和比对统计
  四个Topology v2-lite离线接口，并统一结构化对象请求的异常边界。
- 2026-08-31：运行身份改为进程启动时快照，禁止旧进程随磁盘 HEAD 漂移。
"""
#fastapi后端文件

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

# 把项目根目录加入 sys.path 以支持导入 app.*
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

# 初始化配置与日志
_cfg = Config()
logger = get_logger("BiologyService", "biology.log")
_SERVICE_STARTED_AT = datetime.now(timezone.utc).isoformat()
_SERVICE_RUNTIME_IDENTITY = capture_runtime_identity(
    service_name="biology",
    project_root=root_dir,
    configured_port=_cfg.BIO_API_PORT,
    started_at=_SERVICE_STARTED_AT,
)

# 初始化FastAPI应用
app = FastAPI(title="生命科学模块API接口 - 成员B完成", version="1.0")

# 定义请求体格式（给Agent/前端调用）
class ProteinSequence(BaseModel):
    sequence: str  # 氨基酸序列


class PDBFileRequest(BaseModel):
    """PDB 文件工具的统一请求体。"""

    file_path: str
    n_modes: int = Field(default=10, ge=1, le=20)
    cutoff: float = Field(default=12.0, ge=4.0, le=20.0)
    chain_id: str | None = Field(default=None, max_length=8)


class PDBChainRequest(BaseModel):
    """需要用户显式指定链ID的PDB单链选择请求体。"""

    file_path: str
    chain_id: str = Field(min_length=1, max_length=8)


class FASTAFileRequest(BaseModel):
    """单记录 Protein FASTA 工具请求体。"""

    file_path: str


class FASTARecordSelectionRequest(BaseModel):
    """具名序列集合的精确记录选择请求体。"""

    collection: dict[str, Any]
    record_id: str = Field(min_length=1, max_length=256)


class FASTARecordPairRequest(BaseModel):
    """具名序列集合中两个显式记录的比对请求体。"""

    collection: dict[str, Any]
    record_ids: list[str] = Field(min_length=2, max_length=2)


class PairwiseAlignmentRequest(BaseModel):
    """成对蛋白质比对统计请求体。"""

    alignment: dict[str, Any]


def _format_tool_response(result):
    """把工具返回统一转换为 API 响应，并保留标准错误结构。"""
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
    """统一执行序列工具并记录成功或失败日志。"""
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
    """统一执行 PDB 文件工具，避免接口层泄漏异常。"""
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
    """统一执行 Protein FASTA 文件工具，避免接口层泄漏异常。"""
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
    """统一执行结构化对象工具，日志只记录键名而不泄漏原始序列。"""
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
    """把 Pydantic 请求体验证错误转换成项目统一错误结构。"""
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

# ===================== HTTP 请求日志中间件 =====================
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

# ===================== API接口 =====================
@app.get("/")
def home():
    return {"message": "生命科学分析API服务运行正常", "author": "成员B"}


@app.get("/api/runtime-info")
def runtime_info():
    """返回不含密钥的运行身份，供实验预检核对真实服务。"""
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
    """计算蛋白质平均疏水性 GRAVY 值。"""
    try:
        return _execute_sequence_tool("compute_gravy", data.sequence, bio.compute_gravy)
    except Exception as exc:
        logger.error("compute_gravy 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/compute_secondary_structure_fraction", summary="计算二级结构倾向比例")
def api_compute_secondary_structure_fraction(data: ProteinSequence):
    """计算螺旋、转角和折叠残基的组成倾向比例。"""
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
    """统计20种标准氨基酸的数量和百分比。"""
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
    """计算不稳定性指数并给出经验稳定性判断。"""
    try:
        return _execute_sequence_tool(
            "compute_instability_index", data.sequence, bio.compute_instability_index
        )
    except Exception as exc:
        logger.error("compute_instability_index 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/compute_extinction_coefficient", summary="计算蛋白质消光系数")
def api_compute_extinction_coefficient(data: ProteinSequence):
    """计算280 nm处还原态和氧化态摩尔消光系数。"""
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
    """使用本地毒性蛋白数据库执行高相似性筛查。"""
    try:
        return _execute_sequence_tool(
            "check_protein_toxicity", data.sequence, bio.check_protein_toxicity
        )
    except Exception as exc:
        logger.error("check_protein_toxicity 接口异常: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@app.post("/api/analyze_pdb_structure", summary="分析PDB结构摘要")
def api_analyze_pdb_structure(data: PDBFileRequest):
    """统计PDB中的链、残基、原子、配体、几何和二级结构注释。"""
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
    """从上传的单记录Protein FASTA中提取标准氨基酸序列。"""
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
    """把至少两条合法记录解析成可供下游精确选择的序列集合。"""
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
    """从序列集合按精确record_id选择一条标准氨基酸序列。"""
    return _execute_object_tool(
        "select_fasta_record",
        fasta_collection_tools.select_fasta_record,
        collection=data.collection,
        record_id=data.record_id,
    )


@app.post("/api/align_fasta_records", summary="比对两个FASTA记录")
def api_align_fasta_records(data: FASTARecordPairRequest):
    """对用户显式指定的两个FASTA记录执行全局蛋白质比对。"""
    return _execute_object_tool(
        "align_fasta_records",
        alignment_tools.align_fasta_records,
        collection=data.collection,
        record_ids=data.record_ids,
    )


@app.post("/api/summarize_pairwise_alignment", summary="汇总蛋白质比对统计")
def api_summarize_pairwise_alignment(data: PairwiseAlignmentRequest):
    """从成对蛋白质比对对象计算可解释的质量统计。"""
    return _execute_object_tool(
        "summarize_pairwise_alignment",
        alignment_tools.summarize_pairwise_alignment,
        alignment=data.alignment,
    )


@app.post("/api/extract_pdb_sequences", summary="提取PDB蛋白质链序列")
def api_extract_pdb_sequences(data: PDBFileRequest):
    """从PDB第一模型按链提取氨基酸序列。"""
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
    """只有用户明确提供chain_id时，才从PDB返回一条可供下游使用的序列。"""
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
    """对指定链或残基最多的链计算最低频非零ANM模态。"""
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

# 启动命令
if __name__ == "__main__":
    import uvicorn
    logger.info(f"正在启动生命科学 API 服务，监听端口: {_cfg.BIO_API_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=_cfg.BIO_API_PORT)
