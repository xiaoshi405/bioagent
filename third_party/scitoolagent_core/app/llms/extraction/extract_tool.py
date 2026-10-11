from typing import List, Dict, Optional, Tuple, Any
import json
import re
import ast
import logging
from ..api_clients import make_async_chat, make_async_embedding

logger = logging.getLogger(__name__)

def extract_tool_path(response_text: str) -> List[str]:
    try:
        match = re.search(r"Plan Chain:\s*(\[[^\]]*\])", response_text)
        if match:
            cleaned_tools_string = re.sub(r",\s*//.*?(?=[\],])", "", match.group(1))
            return ast.literal_eval(cleaned_tools_string)
        logger.warning("No tool path found in response.")
        return []
    except (SyntaxError, ValueError, re.error) as e:
        logger.error(f"Error extracting tool path: {e}")
        return []

def extract_answer_parts(answer_text: str) -> Tuple[Optional[str], Optional[str]]:
    is_completed = extract_regex_group(r"IsCompleted:\s*([^\n]+)", answer_text)
    final_answer = extract_regex_group(r"FinalAnswer:\s*(.+)", answer_text, re.DOTALL)
    return is_completed, final_answer

def extract_regex_group(pattern: str, text: str, flags=0) -> Optional[str]:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None

async def parse_json_response(response: str) -> Any:
    try:
        if "```json" in response:
            response = response.strip()[8:-3]
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        raise ValueError("Invalid JSON format in response")

async def extract_tool_info(text: str, **kwargs) -> Dict:
    prompt = (
        "Given the following description of a tool and its relationship to data types and categories within a knowledge graph, "
        "identify and list all entities and their types, as well as relationships among them:\n"
        f"Description: {text}\n"
    )
    response = await make_async_chat(prompt, **kwargs)
    return await parse_json_response(response)

async def get_tool_info_embedding(tool_info: Dict) -> List:
    text = tool_info['functionality']
    embedding = await make_async_embedding(text)
    return [tool_info['id'], embedding]

async def get_all_tool_info_embeddings(tool_infos: List[Dict]) -> List[List]:
    return [await get_tool_info_embedding(info) for info in tool_infos]

async def get_data_type_embedding(data_type: str) -> List:
    embedding = await make_async_embedding(data_type)
    return [data_type, embedding]

async def get_all_data_type_embeddings(data_types: Dict[str, List[str]]) -> List[List]:
    unique_values = list({value for values in data_types.values() for value in values})
    return [await get_data_type_embedding(value) for value in unique_values]

async def extract_associated_datatypes(text: str) -> Dict:
    prompt = (
        "Given a description that contains an input and an output. Both are entity nouns. "
        "Please provide the output in json format:\n"
        f"{text}\n"
    )
    response = await make_async_chat(prompt)
    return await parse_json_response(response)

async def extract_the_most_accurate_data_type(text: str, input: str, output: str) -> Dict:
    prompt = (
        f"Description: {text}. Input: {input}. Output: {output}.\n"
        "Given a description, and Input as well as Output, identify which input or output is most relevant. "
        "Please answer with the original message from Input as well as Output in json format:\n"
    )
    response = await make_async_chat(prompt, top_p=0.1)
    return await parse_json_response(response)
