from typing import List
from app.llms.api_clients import make_async_chat
from app.llms.planning.prompt_templates import (
    generate_plan_input_template,
    generate_output_template,
    generate_tools_plan_template,
    query_next_tool_template,
)
from app.core import Config
async def generate_plan_input(question, current_output, tool_name: List[str], input_type, output_type, model: str, previous_answer: str) -> str:
    prompt = generate_plan_input_template(question, current_output, tool_name, input_type, output_type, previous_answer)
    info = await make_async_chat(prompt, model=model)
    
    answer_prefixes = ["Input:", "input:"]
    extracted_info = None
    for prefix in answer_prefixes:
        if prefix in info:
            start_index = info.index(prefix) + len(prefix)
            extracted_info = info[start_index:].strip()
            break
    
    if extracted_info is None:
        return "Error: Failed to extract the required parameter from the provided information."
    else:
        return extracted_info

async def generate_output(question: str, output: str, model: str) -> str:
    prompt = generate_output_template(question, output)
    return await make_async_chat(prompt, model)

async def generate_tools_plan(question: str, tools_retrieved_list: list, model: str, previous_answer: str) -> list:
    tools_retrieved = '\n'.join(str(tools_info) for tools_info in tools_retrieved_list)
    prompt = generate_tools_plan_template(question, tools_retrieved)
    return await make_async_chat(prompt, model)

async def query_next_tool(question: str, triplet: str, model: str) -> str:
    prompt = query_next_tool_template(question, triplet)
    return await make_async_chat(prompt, model)


    
