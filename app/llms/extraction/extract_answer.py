#负责大模型回复的提取
from typing import List, Dict
import json
import re
import ast
from ..api_clients import make_async_chat, make_async_embedding

#提取planner给出的执行路径
def extract_tool_path(response_text):
    try:
        match = re.search(r"Plan Chain:\s*(\[[^\]]*\])", response_text)
        if match:
            tools_string = match.group(1)
            try:
                cleaned_tools_string = re.sub(r",\s*//.*?(?=[\],])", "", tools_string)
                tool_path = ast.literal_eval(cleaned_tools_string)
                return tool_path
            except (SyntaxError, ValueError) as e:
                print(f"Syntax error in parsing tool path: {e}")
                return None
        else:
            print("No tool path found in response.")
            return None
    except (re.error, Exception) as e:
        print(f"An error occurred: {e}")
        return None

#判断大模型的回答有没有特定的格式，从而判断大模型的任务是否做完
def extract_answer_parts(answer_text):
    is_completed_match = re.search(r"IsCompleted:\s*([^\n]+)", answer_text)
    final_answer_match = re.search(r"FinalAnswer:\s*(.+)", answer_text, re.DOTALL)
    
    if is_completed_match:
        is_completed = is_completed_match.group(1).strip()
    else:
        is_completed = None
        print("is_completed_match: None")
    
    if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
    else:
        final_answer = None
        print("final_answer_match: None")
    
    return is_completed, final_answer

#自动增加新的知识图谱
async def extract_tool_info(text, **kwargs) -> json:
    prompt = "Given the following description of a tool and its relationship to data types and categories within a knowledge graph, identify and list all entities and their types, as well as relationships among them:\n"\
        'Description: '\
        f'{text}\n'

    info = await make_async_chat(prompt, **kwargs)
    if "```json" in info:
        info = json.loads(info.strip()[8:-3])
    else:
        info = json.loads(info)
    return info


async def get_tool_info_embedding(tool_info: dict) -> list:
    text = tool_info['functionality']
    embedding = await make_async_embedding(text)
    return [tool_info['id'], embedding]


async def get_all_tool_info_embeddings(tool_infos: list) -> list:
    embeddings_list = []
    for tool_info in tool_infos:
        embedding = await get_tool_info_embedding(tool_info)
        embeddings_list.append(embedding)
    return embeddings_list


async def get_data_type_embedding(data_type: str) -> list:
    embedding = await make_async_embedding(data_type)
    return [data_type, embedding]


async def get_all_data_type_embeddings(data_types: Dict[str, List[str]]) -> Dict[str, List[float]]:
    values = [value for data_type_list in data_types.values()
              for value in data_type_list]
    values = list(set(values))
    embeddings_list = []
    for value in values:
        embeddings_list.append(await get_data_type_embedding(value))
    return embeddings_list


async def extract_associated_datatypes(text) -> json:
    prompt = "Given a description that contains an input and an output. Both are entity nouns. Please provide the output in json format:\n"\
        f"{text}\n"

    info = await make_async_chat(prompt)
    if "```json" in info:
        info = json.loads(info.strip()[8:-3])
    else:
        info = json.loads(info)
    assert "input" in info and "output" in info and info, "Invalid output format"
    return info


async def extract_the_most_accurate_data_type(text, input, output):
    info = f"Description: {text}. Input: {input}. Output: {output}."
    prompt = "Given a description, and Input as well as Output, identify which input or output is most relevant from Input as well as Output. Please answer with the original message from Input as well as Output. Please output in json format:\n"\
        f"Now, the Description is as follows:\nDescription: {info}\n"

    result = await make_async_chat(prompt, top_p=0.1)
    if "```json" in result:
        result = json.loads(result.strip()[8:-3])
    else:
        result = json.loads(result)
    assert "input" in result and "output" in result, "Invalid output format"
    return result
