# -*- coding: utf-8 -*-
"""Planning and answer prompt templates."""

import json

from ToolsAgent.ToolsFuns.Biology.tool_name_dict import BIOLOGY_TOOLS_DESC

# Use BIOLOGY_TOOLS_DESC as the tool list.

def _build_biology_tool_prompt() -> str:
    try:
        blocks = []
        for tool_name, info in BIOLOGY_TOOLS_DESC.items():
            if info.get("is_internal"):
                continue
            parameters = json.dumps(info.get("parameters", {}), ensure_ascii=False)
            blocks.append(
                f"- {tool_name}：{info.get('description', '')}\n"
                f"  参数：{parameters}"
            )
        return "\n\n".join(blocks)
    except Exception as exc:
        return f"- 工具清单生成失败：{exc}"


BIOLOGY_TOOL_PROMPT = _build_biology_tool_prompt()

BIOLOGY_AGENT_SYSTEM_PROMPT = f"""你是生物蛋白分析Agent，只能调用以下工具，禁止编造参数与功能：

{BIOLOGY_TOOL_PROMPT}

规则：
1. 序列工具必须传入sequence参数，内容必须是标准单字母氨基酸字符串（A/C/D/E/F/G/H/I/K/L/M/N/P/Q/R/S/T/V/W/Y）
2. PDB工具必须使用用户已上传的.pdb文件；不得编造file_path，也不得把氨基酸序列冒充PDB文件。extract_pdb_sequences只返回链序列映射；只有用户明确给出chain_id时才能调用select_pdb_chain得到单条序列
3. FASTA解析必须使用用户已上传的.fasta/.fa/.faa文件；多记录文件先调用parse_multi_protein_fasta。只有用户明确给出record_id时才能调用select_fasta_record，明确给出两个record_id时才能调用align_fasta_records；不得自动选择第一条
4. 不调用 send_get_request / send_post_request / send_expasy_request 等内部工具
5. 想一次性得到基础pI/分子量、信号肽、跨膜区和溶解度结果时，优先调用 protein_analysis；其他属性调用对应原子工具
6. 结果只返回工具输出，不编造未计算的属性
7. Plan Chain 中的工具名必须与上方列表完全一致，不得自行缩写或变形
"""

def generate_plan_input_template(question, current_output, tool_name,
                                  input_type, output_type, previous_answer):
    return f"""
    You are the information extractor, and your task is to extract the incoming parameters from the currently provided information as input to the tool function.

    This is the reference information for you
    question: {question} 
    tool_name: {tool_name}
    input_type:{input_type}
    output_type: {output_type}
    current_output: {current_output}
    previous_answer: {previous_answer}

    current_output indicates the output of an existing tool, and previous_answer indicates the existing answer to the problem, tool_name is the name of the tool function that needs to be used, input_type is the input parameter type of the tool function, and output_type is the output type of the tool function.

    Before extracting parameters, you first need to check "input_type". If "input_type" contains "pair", such as "smiles pair", "protein sequence pair", it means that multiple parameters need to be extracted. Use between different parameters. '.' separated. If "input_type" does not contain "pair", such as "molecule name", "protein name", only one parameter can be extracted.

    Note: When "input_type" does not contain "pair", but the provided Description contains multiple parameters, you must strictly follow the rules, that is, you can only extract one of the parameters, but not multiple parameters. At this time, you need to make a judgment based on the results currently returned in Description. For example, if you analyze that there are parameters A and B, but the processing result of A already exists in Description, you only need to extract B at this time, otherwise extract A. In short, if "input_type" does not contain "pair", then only one of the unprocessed parameters needs to be extracted.

    You need to combine the input type of the tool function and extract the correct input parameters of the tool function from the description document. It is then given strictly as follows:
    Input:your answer

    Note that your answer should strictly match the input parameter type and need not contain other information. For example, if you need to extract information about smiles, your answer should be "input:CCO" and be sure not to include any additional information.
    """


def generate_output_template(question, output):
    return f"""
    Based on the information provided, you need to determine whether my problem has been solved and give an explanation.

    This is my question:
    question:{question}

    This is the reference content: 
    output:{output}

    Please give your answer.

    Your answer should contain two parts. The first part tells me whether the problem has been solved. If it has been solved, give the corresponding answer; if it has not been solved, give the reason.

    Your output should be given strictly in the following format:

    IsCompleted:Completed or Not Completed\n

    FinalAnswer:Your answer\n

    Note: Before answering, you need to think about whether the question involves Security issues. 
    If it does not involve security issues, you should answer normally. 
    Otherwise, you only need to reply the content of the Security Warning, and you do not need to reply other content. And the IsCompleted part should be Completed at this time
    """


def generate_tools_plan_template(question, tools_retrieved):
    return f"""{BIOLOGY_AGENT_SYSTEM_PROMPT}

---
你是专业的任务规划器。现在请根据下方的任务描述和参考信息，制定工具调用链。

你的回答必须严格遵循以下格式：
- 对每个子任务的详细说明
- 每个子任务所需的工具名称
- 按顺序排列所有工具的规划链

工具及其顺序的格式如下：
Plan Chain:
['tool_A', 'tool_B', 'tool_C', ...]

注意：
1. 如果同一工具需要对不同输入使用多次，应在规划链中出现相应次数
2. 你的回答中必须包含 "Plan Chain:"，且工具名必须来自上方列出的已有工具，不得编造
3. Plan Chain 中不需要添加注释
4. Plan Chain 中的工具名必须与上方工具列表中的名称完全一致

任务问题：
{question}

参考信息（知识图谱检索结果）：
{tools_retrieved}

请给出你的规划。
    """


def query_next_tool_template(question, triplet):
    return f"""
    Your task is to tell me what needs to be done to solve this problem based on the existing information.

    I will give you the problem to be solved:
    {question}

    The existing triple information:
    {triplet}

    You need to analyze the problem and the given information and tell me what needs to be done.

    Note:
    1. You only need to answer the steps that need to be done later. You do not need to include the existing triple information or reflect the thinking process. You only need to tell me what needs to be done. The answer should be as concise as possible.
    2. The existing triple information may be the first step to solve this problem, or it may be the subsequent step.
    3. The existing triple information may also be irrelevant to solving this problem. At this time, you need to think about what needs to be done.
    """
