import os
import re
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import load_index_from_storage
import pandas as pd
import logging
from typing import List, Tuple

def load_index_from_storage_dir(persist_dir):
    """Load a knowledge graph index from a graph folder"""
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        kg_index = load_index_from_storage(storage_context)
        print(f"Knowledge Graph Index loaded successfully from {persist_dir}")
        return storage_context, kg_index
    except Exception as e:
        print(f"Failed to load Knowledge Graph Index from {persist_dir}: {e}")
        return None

def construct_question(question, file_path_list, is_add_filename=False):
    if not is_add_filename or not file_path_list:
        return question

    file_extension_dict = {}
    for file_path in file_path_list:
        ext = os.path.splitext(file_path)[1]
        file_extension_dict.setdefault(ext, []).append(os.path.basename(file_path))

    last_file_ext = os.path.splitext(file_path_list[-1])[1]
    cur_file_list = file_extension_dict.get(last_file_ext, [])

    if not cur_file_list:
        return question

    return question + "\nHere is the list of files: " + ", ".join(cur_file_list)


def load_data(file_path: str) -> pd.DataFrame:
    """Load an Excel data file"""
    try:
        df = pd.read_excel(file_path, sheet_name='Tools')
        logging.info("Data loaded successfully from %s", file_path)
        return df
    except Exception as e:
        logging.error("Failed to load data from %s: %s", file_path, e)
        raise

def handle_multiple_entries(name: str, relation: str, entries: str, inverse_relation: str = None) -> List[Tuple[str, str, str]]:
    """Handle multiple inputs/outputs and generate triples"""
    triplets = []
    for entry in entries.split(','):
        entry = entry.strip()
        triplets.append((name, relation, entry))
        if inverse_relation:
            triplets.append((entry, inverse_relation, name))
    return triplets

def create_triplets(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """Create triples from a DataFrame"""
    triplets = []
    for _, row in df.iterrows():
        triplets += [
            (row['Name'], "is a", row['Category']),
            (row['Name'], "has the functionality that", row['Function'].replace("\n", " "))
        ]
        
        # Handle inputs and outputs
        if ',' in row['Input']:
            triplets += handle_multiple_entries(row['Name'], "inputs", row['Input'], "is the input of")
        else:
            triplets.append((row['Name'], "inputs", row['Input']))
            triplets.append((row['Input'], "is the input of", row['Name']))
        
        if ',' in row['Output']:
            triplets += handle_multiple_entries(row['Name'], "outputs", row['Output'], "is the output of")
        else:
            triplets.append((row['Name'], "outputs", row['Output']))
            triplets.append((row['Output'], "is the output of", row['Name']))
        
        triplets.append((row['Name'], "is sourced from", row['Source']))
        
        # Handle security status
        if row['Safety'] == 'Yes':
            triplets.append((row['Name'], "does not need", "Security Check"))
        elif row['Safety'] == 'No':
            triplets.append((row['Name'], "needs", "Security Check"))
    
    logging.info("Triplets created successfully")
    return triplets

def check_tool_in_list(triplets: list, tool_name: str) -> list:
    """
    Checks if a given tool name exists in the triplets list and returns related triplets.

    Args:
        triplets (list): A list of triplets (e.g., [['tool_name', 'relation', 'value'], ...]).
        tool_name (str): The name of the tool to check for.

    Returns:
        list: A list of triplets related to the tool if found, otherwise a message.
    """
    # Filter triplets that match the tool_name
    related_triplets = [triplet for triplet in triplets if triplet[0] == tool_name]
    
    # Return the result based on whether related_triplets were found
    if related_triplets:
        return related_triplets
    else:
        return [f"No information found for tool '{tool_name}'."]

# tool_to_check = "SMILESToCAS"
# res = check_tool_in_list(tools_retrieved, tool_to_check)
# for item in res:
#     print(item)