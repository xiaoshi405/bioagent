import httpx
import logging
from app.core import Config
logger = logging.getLogger(__name__)

def make_simple_dialogs(text, system_prompt=""):
    system_prompt = "You are ChatGPT, a large language model trained by OpenAI.\n"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

async def make_async_request(url, api_key, payload):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=None)
        if response.status_code != 200:
            logger.error(f"Error in request to {url}: Status {response.status_code}, Response {response.text}")
            raise Exception(f"Connection error with status: {response.status_code}")
        
        return response.json()  

def get_api_url(model):
    """Choose the URL of the API"""
    if model.startswith("qwen"):
        return Config().DASHSCOPE_API_BASE
    return Config().OPENAI_API_BASE

async def make_async_chat(text, model, system_prompt="") -> str:
    url = f"{get_api_url(model)}/chat/completions"
    dialogs = make_simple_dialogs(text, system_prompt=system_prompt)
    api_key = Config().DASHSCOPE_API_KEY if model.startswith("qwen") else Config().OPENAI_API_KEY
    payload = {"model": model, "messages": dialogs}

    response = await make_async_request(url, api_key, payload)
    return response["choices"][0]["message"]["content"]

# async def make_async_chat(text, model, config: Config, system_prompt="", **kwargs) -> str:
#     dialogs = make_simple_dialogs(text, system_prompt=system_prompt)

#     if model.startswith("qwen"):  
#         url = f"{config.DASHSCOPE_API_BASE}/chat/completions"
#         response = await make_async_request(url, config.DASHSCOPE_API_KEY, model=model, messages=dialogs, **kwargs)
#         if response.get("status_code", 200) == 200:
#             return response["choices"][0]["message"]["content"]
#         else:
#             raise Exception(f"Failed to get response from Dashscope. Status: {response['status_code']}, Error: {response['text']}")
#     else:
#         url = f"{config.OPENAI_API_BASE}/chat/completions"
#         response = await make_async_request(url, config.OPENAI_API_KEY, model=model, messages=dialogs, **kwargs)
#         if response.get("status_code", 200) == 200:
#             return response["choices"][0]["message"]["content"]
#         else:
#             raise Exception(f"Failed to get response from Dashscope. Status: {response['status_code']}, Error: {response['text']}")
    

async def make_async_embedding(text, model) -> list:
    url = f"{Config().OPENAI_API_BASE}/embeddings"
    payload = {"model": model, "input": text}
    
    response = await make_async_request(url, Config().OPENAI_API_KEY, payload)
    return response["data"][0]["embedding"]
