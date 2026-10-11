import asyncio
import logging
import sys

from scripts.custom_kg_retrievers import KGTableRetriever
from scripts.utils import load_index_from_storage_dir, construct_question
from app.llms.planning import generate_tools_plan, generate_plan_input, generate_output
from app.llms.extraction import extract_tool_path, extract_answer_parts
from app.tools.safety.toxicity_checker import check_toxicity_by_type
from app.tools.safety import extract_smiles_or_sequences
from app.tools import run_tool
from app.core.config import Config

from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

Settings.embed_model = OpenAIEmbedding(model=Config().EMBEDDING_MODEL_NAME)
Settings.llm = OpenAI(temperature=0, model=Config().MODEL_NAME)

async def execute_tool(tool_name, question, input_str, file_path_list, model, kg_table_retriever, total_output):
    current_output = ""
    subgraph = kg_table_retriever.get_tool_subgraph(tool_name, depth=1, limit=1000)
    tool_info = kg_table_retriever.get_tool_info(subgraph)
    input_type = tool_info['input_format']
    output_type = tool_info['output_format']
    needs_security_check = tool_info['requires_security_check']

    while True:
        input_str = await generate_plan_input(question, current_output, tool_name, input_type, output_type, model=model, previous_answer=total_output)
        print(f"input_str: {input_str}")
        if needs_security_check and check_toxicity(input_str):
            return "Security Warning: The input molecules or proteins are toxic, operation stopped."
        
        current_run_result = await run_tool(tool_name, input_str, file_path_list)
        if current_run_result.startswith("Tool function execution error,"):
            continue
        
        if needs_security_check and check_toxicity(current_run_result):
            return "Security Warning: The output molecules or proteins are toxic, operation stopped."
        
        current_output += f"\n\nNow we have run the **{tool_name}** and obtained result\n {current_run_result}."
        break
    return current_output

def check_toxicity(content):
    extracted_items = extract_smiles_or_sequences(content)
    return check_toxicity_by_type(extracted_items)

async def run_SciToolAgent(question, file_path_list, is_add_filename=False):
    question = construct_question(question, file_path_list, is_add_filename)
    storage_context, kg_index = load_index_from_storage_dir(Config().PERSIST_DIR)
    retry_attempts = 3
    responses_content = ""
    total_output = ""
    while retry_attempts > 0:
        try:
            kg_table_retriever = KGTableRetriever(index=kg_index, similarity_top_k=5, graph_store_query_depth=3, max_tools=10)
            
            query_bundle = QueryBundle(query_str=question)
            tools_retrieved = await kg_table_retriever._retrieve(query_bundle)
            tools_plan = await generate_tools_plan(
                question=question, 
                tools_retrieved_list=tools_retrieved, 
                model=Config().MODEL_NAME, 
                previous_answer=total_output, 
            )
            # print(tools_plan)
            tools_path = extract_tool_path(tools_plan)
            # pdb.set_trace()
            if not tools_path:
                retry_attempts -= 1
                continue

            for tool_name in tools_path:
                current_output = await execute_tool(tool_name, question, question, file_path_list, Config().MODEL_NAME, kg_table_retriever, total_output)
                total_output += current_output
                responses_content += current_output
            
            answer_text = await generate_output(question, total_output, model=Config().MODEL_NAME)
            is_completed, current_final_answer = extract_answer_parts(answer_text)
            responses_content += f"\n\n**Task completion status:** {is_completed}\n\n**Final Answer:** {current_final_answer}"

            if is_completed == "Completed":
                break
            else:
                retry_attempts -= 1

        except Exception as e:
            responses_content += f"\nError: {str(e)}\nRetrying..."
            retry_attempts -= 1
            await asyncio.sleep(1)
    
    return responses_content

if __name__ == '__main__':

    question = "Please predict possible retrosynthetic pathways to synthesize CCOC(=O)C, including key intermediates and reagents."
    file_path_list = []
    is_add_filename = False
    # load_dotenv('example.env', override=True)
    # print(os.getenv('OPENAI_API_KEY'))
    # print(os.getenv('OPENAI_API_BASE'))
    response = asyncio.run(run_SciToolAgent(question, file_path_list, is_add_filename))
    print("Response from run_SciToolAgent:", response)