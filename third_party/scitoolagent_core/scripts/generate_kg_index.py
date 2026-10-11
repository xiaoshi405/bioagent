import os
import logging
import sys
from dotenv import load_dotenv
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from utils import load_data, create_triplets
from typing import List, Tuple
from app.core.config import Config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

Settings.embed_model = OpenAIEmbedding(model=Config().EMBEDDING_MODEL_NAME)
Settings.llm = OpenAI(temperature=0, model=Config().MODEL_NAME)

def build_knowledge_graph(triplets: List[Tuple[str, str, str]], persist_dir: str):
    """Build a knowledge graph and save it, while counting the number of entities and triples"""
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    kg_index = KnowledgeGraphIndex([], storage_context=storage_context)

    entities = set()  # Stores unique entities
    for i, triplet in enumerate(triplets):
        # Add triples to graph
        include_embeddings = (triplet[1] == 'has the functionality that')
        kg_index.upsert_triplet(triplet, include_embeddings=include_embeddings)
        
        entities.add(triplet[0])  
        entities.add(triplet[2])  

        if i % 100 == 0:
            logging.info("Processed %d triplets", i)

    kg_index.storage_context.persist(persist_dir=persist_dir)

    print(f"Knowledge graph persisted to {persist_dir}")
    print(f"Total triplets: {len(triplets)}")
    print(f"Total unique entities (nodes): {len(entities)}")


def main():
    # Load data
    df = load_data(Config().DATA_FILE_PATH)

    # Create triples
    triplets = create_triplets(df)

    # Build a knowledge graph
    build_knowledge_graph(triplets, Config().PERSIST_DIR)


if __name__ == "__main__":
    main()
