# -*- coding: utf-8 -*-
"""Retrieve tools, plan workflows and execute requests."""

import asyncio
import json
import os
import time
import traceback

# This channel binds parameters without validation or repair.
from scripts.custom_kg_retrievers import KGTableRetriever
from scripts.utils import load_index_from_storage_dir, construct_question
from scripts.typed_planning_runtime import (
    build_actionable_empty_plan_rejection,
    run_typed_planning,
)
from scripts.baseline_strategies import build_replan_prompt
from scripts.plan_sampling import (
    build_plan_sample_id,
    collect_first_content_response,
    safe_plan_trace,
    sha256_text,
    validate_frozen_plan,
)
from app.llms.planning import generate_tools_plan, generate_output, generate_plan_input
from app.llms.api_clients import make_async_chat_with_metadata
from app.llms.extraction import extract_tool_path
from app.tools import run_tool
from app.tools.tool_runner import find_unknown_tool_names
from app.tools.tool_parameters import build_tool_parameter_context
from app.tools.request_guard import evaluate_explicit_tool_request
from app.core.config import Config
from app.core.logger import get_logger
from llama_index.core import QueryBundle, Settings
from llama_index.llms.openai import OpenAI
from app.core.embedding_runtime import configure_llama_index_embedding

_cfg = Config()
logger = get_logger("AgentService", "agent.log")

os.environ["OPENAI_API_KEY"] = _cfg.API_KEY
os.environ["OPENAI_API_BASE"] = _cfg.API_BASE
os.environ["HF_ENDPOINT"] = _cfg.HF_ENDPOINT
os.environ["LLAMA_INDEX_CACHE_DIR"] = _cfg.LLAMA_INDEX_CACHE_DIR

Settings.llm = OpenAI(
    model=_cfg.LLAMAINDEX_FAKE_MODEL,
    api_key=_cfg.API_KEY,
    api_base=_cfg.API_BASE,
    additional_kwargs={"model": _cfg.MODEL_NAME}
)


def _ensure_embedding_runtime():
    return configure_llama_index_embedding(
        _cfg.BGE_MODEL_PATH,
        cache_dir=_cfg.LLAMA_INDEX_CACHE_DIR,
    )


def _get_tool_io_types(tool_name, triples):
    input_type = "amino acid sequence"
    output_type = "analysis result"
    for triple in triples:
        if len(triple) >= 3 and triple[0] == tool_name:
            if triple[1] == 'inputs':
                input_type = triple[2]
            elif triple[1] == 'outputs':
                output_type = triple[2]
    return input_type, output_type


def _inspect_tool_output(tool_output):
    try:
        if isinstance(tool_output, dict):
            result = tool_output
        elif isinstance(tool_output, str) and tool_output.strip():
            parsed = json.loads(tool_output)
            result = parsed if isinstance(parsed, dict) else {}
        else:
            result = {}
        status = result.get("status")
        if status not in {"success", "error"}:
            status = "error"
        return {
            "status": status,
            "message": result.get("message"),
        }
    except Exception as exc:
        logger.warning("检查工具输出状态失败: %s", exc)
        return {
            "status": "error",
            "message": "工具输出不是可解析的JSON对象",
        }


def _normalize_retrieved_triples(triples):
    try:
        normalized = []
        for triple in triples or []:
            if isinstance(triple, (list, tuple)):
                normalized.append([str(value) for value in triple])
        return normalized
    except Exception as exc:
        logger.warning("规范化检索三元组失败: %s", exc)
        return []


def _retrieved_tool_names(triples):
    try:
        names = []
        for triple in triples or []:
            if (
                isinstance(triple, (list, tuple))
                and len(triple) >= 3
                and triple[1] == "has the functionality that"
                and str(triple[0]) not in names
            ):
                names.append(str(triple[0]))
        return names
    except Exception as exc:
        logger.warning("提取检索工具名失败: %s", exc)
        return []


async def _capture_plan_from_retrieved(
    *,
    question,
    tools_retrieved,
    experiment_id,
    question_id,
    repeat_index,
    model,
    planner_callable=generate_tools_plan,
    config=None,
):
    cfg = config if config is not None else _cfg
    max_attempts = int(
        getattr(cfg, "EXPERIMENT_PLAN_NETWORK_MAX_ATTEMPTS", 3)
    )
    max_attempts = min(5, max(1, max_attempts))
    backoff_seconds = float(
        getattr(cfg, "EXPERIMENT_PLAN_NETWORK_BACKOFF_SECONDS", 1.0)
    )
    backoff_seconds = min(30.0, max(0.0, backoff_seconds))
    normalized_triples = _normalize_retrieved_triples(tools_retrieved)
    async def invoke_planner():
        return await planner_callable(
            question=question,
            tools_retrieved_list=tools_retrieved,
            model=model,
            previous_answer="",
        )

    sample = await collect_first_content_response(
        planner_callable=invoke_planner,
        parser_callable=extract_tool_path,
        unknown_tool_callable=find_unknown_tool_names,
        max_network_attempts=max_attempts,
        backoff_seconds=backoff_seconds,
    )
    if sample.get("status") != "success":
        return {
            "status": "error",
            "message": sample.get("message"),
            "issues": [
                {
                    "code": sample.get("code"),
                    "message": sample.get("message"),
                    "attempts": sample.get("attempts") or [],
                }
            ],
        }
    response_sha256 = sample["response_sha256"]
    frozen_plan = {
        "schema_version": "1.0",
        "record_status": "captured",
        "experiment_id": str(experiment_id or "").strip(),
        "question_id": str(question_id or "").strip(),
        "repeat_index": int(repeat_index),
        "model": str(model or "").strip(),
        "plan_sample_id": build_plan_sample_id(
            experiment_id=experiment_id,
            question_id=question_id,
            repeat_index=repeat_index,
            model=model,
            response_sha256=response_sha256,
        ),
        "capture_outcome": sample["capture_outcome"],
        "selected_attempt_index": sample["selected_attempt_index"],
        "response_sha256": response_sha256,
        "planned_chain": sample["planned_chain"],
        "unknown_tools": sample["unknown_tools"],
        "retrieved_triples": normalized_triples,
        "retrieved_tool_names": _retrieved_tool_names(normalized_triples),
        "attempts": sample["attempts"],
    }
    validation = validate_frozen_plan(frozen_plan)
    return (
        {"status": "success", "frozen_plan": validation["frozen_plan"]}
        if validation.get("status") == "success"
        else validation
    )


async def capture_SciToolAgent_plan(
    question,
    file_path_list,
    is_add_filename=False,
    *,
    experiment_id="",
    question_id="",
    repeat_index=0,
    model=None,
):
    original_question = question
    request_guard = evaluate_explicit_tool_request(original_question)
    if request_guard.get("decision") == "reject":
        raw_response = json.dumps(request_guard, ensure_ascii=False)
        response_sha256 = sha256_text(raw_response)
        attempt = {
            "attempt_index": 1,
            "attempt_kind": "content",
            "outcome": "request_guard_reject",
            "exception_type": None,
            "response_sha256": response_sha256,
            "raw_response": raw_response,
            "parsed_chain": [],
            "unknown_tools": list(request_guard.get("unknown_tools") or []),
            "latency_ms": 0.0,
        }
        frozen_plan = {
            "schema_version": "1.0",
            "record_status": "captured",
            "experiment_id": str(experiment_id or "").strip(),
            "question_id": str(question_id or "").strip(),
            "repeat_index": int(repeat_index),
            "model": str(model or _cfg.MODEL_NAME).strip(),
            "plan_sample_id": build_plan_sample_id(
                experiment_id=experiment_id,
                question_id=question_id,
                repeat_index=repeat_index,
                model=model or _cfg.MODEL_NAME,
                response_sha256=response_sha256,
            ),
            "capture_outcome": "request_guard_reject",
            "selected_attempt_index": 1,
            "response_sha256": response_sha256,
            "planned_chain": [],
            "unknown_tools": attempt["unknown_tools"],
            "retrieved_triples": [],
            "retrieved_tool_names": [],
            "attempts": [attempt],
        }
        validation = validate_frozen_plan(frozen_plan)
        return (
            {"status": "success", "frozen_plan": validation["frozen_plan"]}
            if validation.get("status") == "success"
            else validation
        )

    prepared_question = construct_question(
        question,
        file_path_list,
        is_add_filename,
    )
    embedding_result = _ensure_embedding_runtime()
    if embedding_result.get("status") != "success":
        return {
            "status": "error",
            "message": embedding_result.get("message", "嵌入模型不可用"),
            "issues": list(embedding_result.get("issues") or []),
        }
    try:
        _, kg_index = load_index_from_storage_dir(_cfg.PERSIST_DIR)
        kg_table_retriever = KGTableRetriever(
            index=kg_index,
            similarity_top_k=5,
            graph_store_query_depth=3,
            max_tools=10,
        )
        tools_retrieved = await kg_table_retriever._retrieve(
            QueryBundle(query_str=prepared_question)
        )
        if not tools_retrieved:
            return {
                "status": "error",
                "message": "未检索到可用于规划的工具",
                "issues": [
                    {
                        "code": "PLAN_CAPTURE_RETRIEVAL_FAILURE",
                        "message": "未检索到可用于规划的工具",
                    }
                ],
            }
    except Exception as exc:
        logger.error("实验计划采样的索引或检索阶段失败", exc_info=True)
        return {
            "status": "error",
            "message": "计划采样的索引或检索阶段失败",
            "issues": [
                {
                    "code": "PLAN_CAPTURE_RETRIEVAL_FAILURE",
                    "message": "计划采样的索引或检索阶段失败",
                    "exception_type": type(exc).__name__,
                }
            ],
        }
    return await _capture_plan_from_retrieved(
        question=prepared_question,
        tools_retrieved=tools_retrieved,
        experiment_id=experiment_id,
        question_id=question_id,
        repeat_index=repeat_index,
        model=model or _cfg.MODEL_NAME,
        config=_cfg,
    )


async def capture_SciToolAgent_replan(
    *,
    question,
    original_plan_chain,
    available_input_types,
    type_issues,
    model,
    max_output_tokens,
    required_tool_names=None,
):
    started = time.perf_counter()
    try:
        prompt_result = build_replan_prompt(
            question=question,
            original_plan_chain=original_plan_chain,
            available_input_types=available_input_types,
            type_issues=type_issues,
            required_tool_names=required_tool_names,
        )
        if prompt_result.get("status") != "success":
            return prompt_result
        response = await make_async_chat_with_metadata(
            prompt_result["prompt"],
            str(model or _cfg.MODEL_NAME),
            max_output_tokens=max_output_tokens,
        )
        raw_response = str(response.get("content") or "")
        return {
            "status": "success",
            "raw_response": raw_response,
            "planned_chain": extract_tool_path(raw_response),
            "usage": dict(response.get("usage") or {}),
            "latency_ms": max(
                0.0,
                (time.perf_counter() - started) * 1000.0,
            ),
        }
    except Exception as exc:
        logger.error("L1单次重规划请求失败: %s", exc, exc_info=True)
        return {
            "status": "error",
            "message": "L1单次重规划请求失败",
            "issues": [
                {
                    "code": "LLM_REPLAN_REQUEST_FAILURE",
                    "message": "L1单次重规划请求失败",
                    "exception_type": type(exc).__name__,
                }
            ],
            "latency_ms": max(
                0.0,
                (time.perf_counter() - started) * 1000.0,
            ),
        }


async def run_SciToolAgent(
    question,
    file_path_list,
    is_add_filename=False,
    *,
    frozen_plan=None,
):
    original_question = question
    tool_parameter_context = build_tool_parameter_context(original_question)

    reasoning = {
        "retrieved_tools": [],
        "plan_chain": [],
        "tool_steps": [],
        "typed_planning": None,
        "request_guard": None,
        "plan_sampling": safe_plan_trace(frozen_plan),
        "execution_summary": {
            "completed": False,
            "attempted_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "error_type": None,
        },
    }

    # Apply the explicit-tool guard in all experiment conditions.
    request_guard = evaluate_explicit_tool_request(original_question)
    reasoning["request_guard"] = request_guard
    if request_guard.get("decision") == "reject":
        unknown_tools = list(request_guard.get("unknown_tools") or [])
        detail_code = request_guard.get("detail_code") or "REQUEST_GUARD_ERROR"
        reasoning["execution_summary"] = {
            "completed": False,
            "attempted_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "error_type": "capability_failure",
            "detail_code": detail_code,
        }
        if unknown_tools:
            tool_text = "、".join(unknown_tools)
            user_message = (
                f"工具 {tool_text} 未在当前系统注册，无法调用。"
                "系统不会使用其他工具冒充，也不会编造其结果。"
            )
        else:
            user_message = "工具请求安全检查失败，系统未进入规划或执行阶段。"
        error_payload = {
            "status": "error",
            "message": user_message,
            "error_type": "capability_failure",
            "detail_code": detail_code,
            "unknown_tools": unknown_tools,
            "required_user_inputs": [],
        }
        logger.warning(
            "[Step 0] 显式工具请求被规划前门禁拦截 | "
            "detail_code=%s | unknown_count=%s",
            detail_code,
            len(unknown_tools),
        )
        return {
            "answer": (
                f"{json.dumps(error_payload, ensure_ascii=False)}"
                "\n\n**Final Answer:**\n"
                f"问题未完成：{user_message}"
            ),
            "reasoning": reasoning,
        }

    question = construct_question(question, file_path_list, is_add_filename)
    persist_dir = _cfg.PERSIST_DIR

    frozen_validation = (
        validate_frozen_plan(frozen_plan)
        if frozen_plan is not None
        else None
    )
    if (
        isinstance(frozen_validation, dict)
        and frozen_validation.get("status") != "success"
    ):
        reasoning["execution_summary"]["error_type"] = "invalid_frozen_plan"
        return {
            "status": "error",
            "error_code": "INVALID_FROZEN_PLAN",
            "answer": "错误：实验冻结计划校验失败，未执行任何工具。",
            "reasoning": reasoning,
        }
    active_frozen_plan = (
        frozen_validation.get("frozen_plan")
        if isinstance(frozen_validation, dict)
        else None
    )

    if active_frozen_plan is None:
        embedding_result = _ensure_embedding_runtime()
        if embedding_result.get("status") != "success":
            issue = (embedding_result.get("issues") or [{}])[0]
            reasoning["execution_summary"]["error_type"] = (
                "embedding_model_invalid"
            )
            return {
                "status": "error",
                "error_code": issue.get(
                    "code", "EMBEDDING_MODEL_INVALID"
                ),
                "answer": "错误：本地嵌入模型不可用，未进入知识图谱检索。",
                "reasoning": reasoning,
            }
        try:
            logger.info(f"[Step 1] 正在加载索引目录: {persist_dir}")
            storage_context, kg_index = load_index_from_storage_dir(persist_dir)
            logger.info("索引加载成功")
        except Exception as e:
            logger.error("无法加载索引文件", exc_info=True)
            reasoning["execution_summary"]["error_type"] = "index_load_failed"
            return {
                "status": "error",
                "error_code": "INDEX_LOAD_FAILED",
                "answer": f"错误：无法加载索引文件。详情: {e}",
                "reasoning": reasoning,
            }

    retry_attempts = 1 if active_frozen_plan is not None else 3
    total_output = ""
    responses_content = ""
    last_retry_reason = None

    while retry_attempts > 0:
        try:
            if active_frozen_plan is not None:
                tools_retrieved = active_frozen_plan.get(
                    "retrieved_triples",
                    [],
                )
                reasoning["retrieved_tools"] = list(
                    active_frozen_plan.get("retrieved_tool_names") or []
                )
                tools_path = list(
                    active_frozen_plan.get("planned_chain") or []
                )
                logger.info(
                    "[Step 2-3] 重放冻结计划 | plan_sample_id=%s",
                    active_frozen_plan.get("plan_sample_id"),
                )
            else:
                logger.info("[Step 2] 正在启动混合检索...")
                kg_table_retriever = KGTableRetriever(
                    index=kg_index,
                    similarity_top_k=5,
                    graph_store_query_depth=3,
                    max_tools=10,
                )
                query_bundle = QueryBundle(query_str=question)
                tools_retrieved = await kg_table_retriever._retrieve(
                    query_bundle
                )
                logger.info(
                    "获取到相关工具的三元组数量: %s",
                    len(tools_retrieved),
                )
                if not tools_retrieved:
                    logger.warning("未检索到匹配的工具")
                    return {
                        "answer": "当前库中未检索到匹配的工具，请检查您的查询或图谱索引。",
                        "reasoning": reasoning,
                    }
                reasoning["retrieved_tools"] = _retrieved_tool_names(
                    tools_retrieved
                )
                logger.info("[Step 3] 正在生成工具执行计划...")
                tools_plan = await generate_tools_plan(
                    question=question,
                    tools_retrieved_list=tools_retrieved,
                    model=_cfg.MODEL_NAME,
                    previous_answer=total_output,
                )
                tools_path = extract_tool_path(tools_plan)
            if not tools_path:
                logger.warning("模型计划无法解析；冻结计划不会触发内容重抽")
                last_retry_reason = "EMPTY_OR_UNPARSEABLE_PLAN"
                retry_attempts -= 1
                continue

            reasoning["plan_chain"] = tools_path

            unknown_tools = find_unknown_tool_names(tools_path)
            if unknown_tools and not _cfg.TYPE_PLANNING_ENABLED:
                reasoning["execution_summary"] = {
                    "completed": False,
                    "attempted_steps": 0,
                    "successful_steps": 0,
                    "failed_steps": 0,
                    "error_type": "unknown_tool_plan",
                }
                error_payload = {
                    "status": "error",
                    "message": "计划包含当前注册表中不存在的工具，已拒绝执行",
                    "error_type": "unknown_tool_plan",
                    "unknown_tool_count": len(unknown_tools),
                }
                logger.warning(
                    "[Step 3] B0闭集工具检查拒绝计划 | unknown_count=%s",
                    len(unknown_tools),
                )
                return {
                    "answer": (
                        f"{json.dumps(error_payload, ensure_ascii=False)}"
                        "\n\n**Final Answer:**\n"
                        "问题未完成：计划引用了当前系统未注册的工具，"
                        "因此未执行任何工具。"
                    ),
                    "reasoning": reasoning,
                }

            # Use the legacy executor when typed planning is disabled.
            if _cfg.TYPE_PLANNING_ENABLED:
                logger.info(
                    "[Step 3.5] 类型约束规划已开启，构建初始Artifact并执行结构化计划"
                )
                typed_result = await run_typed_planning(
                    tool_path=tools_path,
                    question=original_question,
                    file_path_list=file_path_list,
                    config=_cfg,
                    tool_parameters=tool_parameter_context,
                )
                reasoning["typed_planning"] = typed_result.get(
                    "safe_trace",
                    {},
                )

                if typed_result.get("status") != "success":
                    error_message = typed_result.get(
                        "message",
                        "类型约束规划或执行失败",
                    )
                    failed_execution = typed_result.get("execution") or {}
                    executed_steps = failed_execution.get(
                        "executed_steps",
                        [],
                    )
                    attempted_steps = len(
                        failed_execution.get("tool_results") or []
                    )
                    reasoning["execution_summary"] = {
                        "completed": False,
                        "attempted_steps": attempted_steps,
                        "successful_steps": len(executed_steps),
                        "failed_steps": max(
                            0,
                            attempted_steps - len(executed_steps),
                        ),
                        "error_type": typed_result.get("error_type"),
                        "detail_code": typed_result.get("detail_code"),
                    }
                    error_payload = {
                        "status": "error",
                        "message": error_message,
                        "error_type": typed_result.get("error_type"),
                        "detail_code": typed_result.get("detail_code"),
                        "issues": typed_result.get("issues") or [],
                        "required_user_inputs": typed_result.get(
                            "required_user_inputs"
                        ) or [],
                    }
                    logger.warning(
                        "[Step 3.5] 类型化分支阻断下游总结 | error_type=%s | message=%s",
                        typed_result.get("error_type"),
                        error_message,
                    )
                    final_answer = (
                        f"{json.dumps(error_payload, ensure_ascii=False)}"
                        "\n\n**Final Answer:**\n"
                        f"问题未完成：{error_message}"
                    )
                    return {
                        "answer": final_answer,
                        "reasoning": reasoning,
                    }

                planning_result = typed_result.get("planning") or {}
                execution_result = typed_result.get("execution") or {}
                reasoning["plan_chain_executed"] = planning_result.get(
                    "plan_after",
                    tools_path,
                )
                reasoning["tool_steps"] = execution_result.get(
                    "executed_steps",
                    [],
                )
                successful_steps = len(reasoning["tool_steps"])
                reasoning["execution_summary"] = {
                    "completed": True,
                    "attempted_steps": successful_steps,
                    "successful_steps": successful_steps,
                    "failed_steps": 0,
                    "error_type": None,
                }

                typed_total_output = ""
                typed_responses_content = ""
                for item in execution_result.get("tool_results") or []:
                    tool_name = item.get("tool_name", "unknown_tool")
                    current_output = json.dumps(
                        item.get("result") or {},
                        ensure_ascii=False,
                    )
                    typed_total_output += (
                        f"\n工具执行结果 ({tool_name}):\n{current_output}"
                    )
                    typed_responses_content += current_output

                logger.info("[Step 5] 正在总结v0.3b类型化工具结果...")
                answer_text = await generate_output(
                    question,
                    typed_total_output,
                    model=_cfg.MODEL_NAME,
                )
                final_answer = (
                    f"{typed_responses_content}"
                    f"\n\n**Final Answer:**\n{answer_text}"
                )
                return {
                    "answer": final_answer,
                    "reasoning": reasoning,
                }

            previous_tool_output = ""
            baseline_step_statuses = []
            for i, tool_name in enumerate(tools_path):
                logger.info(f"[Step 4] 正在调用本地工具 ({i+1}/{len(tools_path)}): {tool_name}")

                # The first tool receives the original request.
                # Later tools use model-extracted parameters from prior output.
                if i > 0 and previous_tool_output:
                    input_type, output_type = _get_tool_io_types(tool_name, tools_retrieved)
                    logger.info(f"[Step 4.1] 正在让 LLM 提取 {tool_name} 的输入参数 (input_type={input_type})...")
                    try:
                        extracted_args = await generate_plan_input(
                            question=question,
                            current_output=previous_tool_output,
                            tool_name=[tool_name],
                            input_type=input_type,
                            output_type=output_type,
                            model=_cfg.MODEL_NAME,
                            previous_answer=total_output,
                        )
                        tool_args = extracted_args
                        logger.info(f"LLM 提取的参数: {tool_args[:120]}...")
                    except Exception as e:
                        logger.warning(f"LLM 参数提取失败，回退使用原始问题: {e}")
                        tool_args = question
                else:
                    tool_args = question

                current_output = await run_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    file_path_list=file_path_list if 'file_path_list' in locals() else [],
                    previous_tool_output=previous_tool_output,
                    tool_parameters=tool_parameter_context,
                )

                previous_tool_output = current_output
                total_output += f"\n工具执行结果 ({tool_name}):\n{current_output}"
                responses_content += current_output
                output_status = _inspect_tool_output(current_output)
                baseline_step_statuses.append(output_status)

                reasoning["tool_steps"].append({
                    "tool_name": tool_name,
                    "args_preview": tool_args[:200] if len(tool_args) > 200 else tool_args,
                    "output_preview": current_output[:500] if len(current_output) > 500 else current_output,
                    "status": output_status["status"],
                    "error_message": output_status.get("message"),
                })

            failed_steps = sum(
                item.get("status") != "success"
                for item in baseline_step_statuses
            )
            reasoning["execution_summary"] = {
                "completed": (
                    len(baseline_step_statuses) == len(tools_path)
                    and failed_steps == 0
                ),
                "attempted_steps": len(baseline_step_statuses),
                "successful_steps": len(baseline_step_statuses) - failed_steps,
                "failed_steps": failed_steps,
                "error_type": (
                    "tool_failure" if failed_steps > 0 else None
                ),
            }
            logger.info("[Step 5] 正在生成最终答案...")
            answer_text = await generate_output(
                question,
                total_output,
                model=_cfg.MODEL_NAME
            )
            final_answer = f"{responses_content}\n\n**Final Answer:**\n{answer_text}"
            return {"answer": final_answer, "reasoning": reasoning}

        except Exception as e:
            logger.error("系统运行崩溃，发生异常", exc_info=True)
            last_retry_reason = "AGENT_INTERNAL_EXCEPTION"
            retry_attempts -= 1
            await asyncio.sleep(1)

    if last_retry_reason == "EMPTY_OR_UNPARSEABLE_PLAN":
        typed_result = None
        if _cfg.TYPE_PLANNING_ENABLED:
            typed_result = await run_typed_planning(
                tool_path=[],
                question=original_question,
                file_path_list=file_path_list,
                config=_cfg,
                tool_parameters=tool_parameter_context,
            )
            reasoning["typed_planning"] = typed_result.get(
                "safe_trace",
                {},
            )
        rejection = build_actionable_empty_plan_rejection(typed_result)
        error_payload = rejection.get("payload") or {
            "status": "error",
            "message": "空计划已拒绝执行",
            "error_type": "empty_or_unparseable_plan",
        }
        reasoning["execution_summary"] = {
            "completed": False,
            "attempted_steps": 0,
            "successful_steps": 0,
            "failed_steps": 0,
            "error_type": "empty_or_unparseable_plan",
            "detail_code": error_payload.get("detail_code"),
        }
        return {
            "answer": (
                f"{json.dumps(error_payload, ensure_ascii=False)}"
                "\n\n**Final Answer:**\n"
                f"{rejection.get('final_message')}"
            ),
            "reasoning": reasoning,
        }

    reasoning["execution_summary"]["completed"] = False
    reasoning["execution_summary"]["error_type"] = (
        "agent_retries_exhausted"
    )
    return {
        "status": "error",
        "error_code": "AGENT_RETRIES_EXHAUSTED",
        "answer": "抱歉，系统多次尝试后未能完成任务。",
        "reasoning": reasoning,
    }
