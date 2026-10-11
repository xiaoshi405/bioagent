"""KG Retrievers."""

import pdb
import logging,re
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.keyword_table.utils import (
    extract_keywords_given_response,
)
from llama_index.core.indices.knowledge_graph.base import KnowledgeGraphIndex
from llama_index.core.indices.query.embedding_utils import get_top_k_embeddings
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import BasePromptTemplate, PromptTemplate, PromptType
from llama_index.core.prompts.default_prompts import (
    DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE,
)
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.service_context import ServiceContext
from llama_index.core.settings import Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.utils import print_text, truncate_text
# from app.llms import query_next_tool

DQKET = DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE
DEFAULT_NODE_SCORE = 1000.0
GLOBAL_EXPLORE_NODE_LIMIT = 3
REL_TEXT_LIMIT = 1000

logger = logging.getLogger(__name__)

Embedding = List[float]

class KGRetrieverMode(str, Enum):
    """Query mode enum for Knowledge Graphs.

    Can be passed as the enum struct, or as the underlying string.

    Attributes:
        KEYWORD ("keyword"): Default query mode, using keywords to find triplets.
        EMBEDDING ("embedding"): Embedding mode, using embeddings to find
            similar triplets.
        HYBRID ("hybrid"): Hybrid mode, combining both keywords and embeddings
            to find relevant triplets.
    """

    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


class KGTableRetriever(BaseRetriever):
    """KG Table Retriever.

    Arguments are shared among subclasses.

    Args:
        query_keyword_extract_template (Optional[QueryKGExtractPrompt]): A Query
            KG Extraction
            Prompt (see :ref:`Prompt-Templates`).
        refine_template (Optional[BasePromptTemplate]): A Refinement Prompt
            (see :ref:`Prompt-Templates`).
        text_qa_template (Optional[BasePromptTemplate]): A Question Answering Prompt
            (see :ref:`Prompt-Templates`).
        max_keywords_per_query (int): Maximum number of keywords to extract from query.
        num_chunks_per_query (int): Maximum number of text chunks to query.
        include_text (bool): Use the document text source from each relevant triplet
            during queries.
        retriever_mode (KGRetrieverMode): Specifies whether to use keywords,
            embeddings, or both to find relevant triplets. Should be one of "keyword",
            "embedding", or "hybrid".
        similarity_top_k (int): The number of top embeddings to use
            (if embeddings are used).
        graph_store_query_depth (int): The depth of the graph store query.
        use_global_node_triplets (bool): Whether to get more keywords(entities) from
            text chunks matched by keywords. This helps introduce more global knowledge.
            While it's more expensive, thus to be turned off by default.
        max_knowledge_sequence (int): The maximum number of knowledge sequence to
            include in the response. By default, it's 30.
    """

    def __init__(
        self,
        index: KnowledgeGraphIndex,
        llm: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        query_keyword_extract_template: Optional[BasePromptTemplate] = None,
        max_keywords_per_query: int = 10,
        num_chunks_per_query: int = 10,
        include_text: bool = True,
        retriever_mode: Optional[KGRetrieverMode] = KGRetrieverMode.KEYWORD,
        similarity_top_k: int = 2,
        graph_store_query_depth: int = 2,
        use_global_node_triplets: bool = False,
        max_knowledge_sequence: int = REL_TEXT_LIMIT,
        max_tools:int=10,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        assert isinstance(index, KnowledgeGraphIndex)
        self._index = index
        self._index_struct = self._index.index_struct
        self._docstore = self._index.docstore

        self.max_keywords_per_query = max_keywords_per_query
        self.num_chunks_per_query = num_chunks_per_query
        self.query_keyword_extract_template = query_keyword_extract_template or DQKET
        self.similarity_top_k = similarity_top_k
        self._include_text = include_text
        self._retriever_mode = (
            KGRetrieverMode(retriever_mode)
            if retriever_mode
            else KGRetrieverMode.KEYWORD
        )

        self._llm = llm or Settings.llm
        self._embed_model = embed_model or Settings.embed_model
        self.max_tools = max_tools
        self._graph_store = index.graph_store
        self.graph_store_query_depth = graph_store_query_depth
        self.use_global_node_triplets = use_global_node_triplets
        self.max_knowledge_sequence = max_knowledge_sequence
        self._verbose = kwargs.get("verbose", False)
        refresh_schema = kwargs.get("refresh_schema", False)
        try:
            self._graph_schema = self._graph_store.get_schema(refresh=refresh_schema)
        except NotImplementedError:
            self._graph_schema = ""
        except Exception as e:
            logger.warning(f"Failed to get graph schema: {e}")
            self._graph_schema = ""
        super().__init__(
            callback_manager=callback_manager or Settings.callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    def extract_function_triple(self, triplets, function_name='Has the functionality that'):
        tri_list = []
        for tri in triplets:
            # if tri.__contains__(function_name):
            #     tri_list.append(tri)
            if function_name in tri:
                tri_list.append(tri)

        return tri_list

    def triple2text(self, triplets):
        text = ""
        for t in triplets:
            text = text + t[0]+" "+t[1]+" "+t[2]+". "
        return text

    def list2string(self, triplets_list):
        tri_list = []   
        for triplets in triplets_list:
            tri_list.append("('"+triplets[0]+"', '"+triplets[1]+"', '"+triplets[2]+"')")
            
        return tri_list

    def batch_get_text_embeddings(self, text_list: List[str], max_workers: int = 5) -> List[Embedding]:
        """
        Process a list of texts and get their embeddings using the get_text_embedding method in a multithreaded way.

        Args:
            text_list (List[str]): A list of text strings to be embedded.
            max_workers (int): The maximum number of threads to use.

        Returns:
            List[Embedding]: A list of embeddings corresponding to the input texts.
        """
        embeddings = [None] * len(text_list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self._embed_model.get_text_embedding, text): i for i, text in enumerate(text_list)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    embeddings[index] = future.result()
                except Exception as exc:
                    print(f'Text at index {index} generated an exception: {exc}')
        return embeddings

    # async def _retrieve(
    #     self,
    #     query_bundle: QueryBundle):
    #
    #     # compute embedding similarity between all tools with query
    #     query_embedding = self._embed_model.get_text_embedding(
    #         query_bundle.query_str
    #     )
    #     all_rel_texts = list(self._index_struct.embedding_dict.keys())
    #     fun_rel_texts = self.extract_function_triple(all_rel_texts, function_name='functionality')
    #     rel_text_embeddings = [
    #         self._index_struct.embedding_dict[_id] for _id in fun_rel_texts
    #     ]
    #     similarities, top_rel_texts = get_top_k_embeddings(
    #         query_embedding,
    #         rel_text_embeddings,
    #         similarity_top_k=self.similarity_top_k,
    #         embedding_ids=fun_rel_texts,
    #     )
    #
    #     # get tool name
    #     tool_retrieved = []
    #     for text in top_rel_texts:
    #         item = re.search(r"(?<=\(')(.*?)(?=\',)", text)
    #         item = item.group(1)
    #         tool_retrieved.append(item)
    #     # print(f"tool_retrieved:{tool_retrieved}")
    #
    #     # get tool subgraph
    #     tool_subgraph = self._graph_store.get_rel_map(
    #         tool_retrieved, depth=self.graph_store_query_depth,limit=self.max_knowledge_sequence
    #     )
    #     # print(f"tool_subgraph:{tool_subgraph}")
    #     # pdb.set_trace()
    #
    #     # compute embedding similarity between subgraph'tools with query+previous_tool
    #     c = 0
    #     for tool in tool_subgraph:
    #         fun_rel_texts = self.extract_function_triple(self.list2string(tool_subgraph[tool]), function_name='functionality')
    #         fun_rel_texts = list(dict.fromkeys(fun_rel_texts)) #remove deplicate
    #         # tool_existed = "There is a computational tool represented by a triplet:" + fun_rel_texts[0]
    #         fun_rel_texts = fun_rel_texts[1:]
    #         if fun_rel_texts == []:
    #             continue
    #
    #         tool_existed = self._graph_store.get_rel_map([tool], 1, limit=self.max_knowledge_sequence)[tool]
    #         tool_existed = self.triple2text(tool_existed[1:])
    #         find_next_tools = "Query: "+query_bundle.query_str + "\n The existed tool: "+ tool_existed + "\n According to the query and the existed tool, please imagine what next tools do we need, only show the functionality of the required tool in a sentence. DO NOT give any explanations."
    #         next_tools = self._llm.predict(PromptTemplate(find_next_tools))
    #         # print(f"next_tools:\n{next_tools}")
    #         # new_query = await query_next_tool(query_bundle.query_str, fun_rel_texts[0], model='gpt-4o')
    #         # print(f"new_query:\n{new_query}")
    #         query_embedding = self._embed_model.get_text_embedding(next_tools)
    #
    #         # query_embedding = self._embed_model.get_text_embedding(query_bundle.query_str+tool_existed)
    #
    #         rel_text_embeddings = [
    #             self._index_struct.embedding_dict[_id] for _id in fun_rel_texts
    #         ]
    #
    #         # for _id in fun_rel_texts:
    #         #     if _id in self._index_struct.embedding_dict:
    #         #         rel_text_embeddings.append(self._index_struct.embedding_dict[_id])
    #         #     else:
    #         #         print(f"Warning: No embedding found for {_id}, skipping.")
    #
    #         similarities_sub, top_rel_texts_sub = get_top_k_embeddings(
    #             query_embedding,
    #             rel_text_embeddings,
    #             similarity_top_k=self.similarity_top_k, # 3
    #             embedding_ids=fun_rel_texts,
    #         )
    #         for text in top_rel_texts_sub:
    #             item = re.search(r"(?<=\(')(.*?)(?=\',)", text)
    #             item = item.group(1)
    #             tool_retrieved.append(item)
    #         similarities = similarities + [i * similarities[c] for i in similarities_sub]
    #         c = c+1
    #
    #     #  remove deplicate
    #     tool_retrieved_dep = list(dict.fromkeys(tool_retrieved))
    #     # get tool weights based on similarity
    #     tool_weights = {}
    #     for item in tool_retrieved_dep:
    #         weight = [similarities[i] for i, x in enumerate(tool_retrieved) if x == item]
    #         tool_weights[item]=sum(weight)
    #
    #     # Priority is given to larger weights
    #     tool_retrieved_sort = sorted(tool_retrieved_dep, key=lambda x: tool_weights[x], reverse=True)
    #     tool_retrieved_sort = tool_retrieved_sort[:self.max_tools]
    #     # get tool node
    #     tool_subgraph = self._graph_store.get_rel_map(
    #         tool_retrieved_sort, 1, limit=self.max_knowledge_sequence
    #     )
    #     print(tool_retrieved_sort)
    #     # convert to text
    #     # tool_text = []
    #     # for tool in tool_retrieved_sort:
    #     #     tool_text.append(self.triple2text(tool_subgraph[tool]))
    #     original_triples = []
    #     for tool in tool_retrieved_sort:
    #         original_triples.extend(tool_subgraph[tool])
    #
    #     return original_triples

    async def _retrieve(
            self,
            query_bundle: QueryBundle):

        # compute embedding similarity between all tools with query
        query_embedding = self._embed_model.get_text_embedding(
            query_bundle.query_str
        )
        all_rel_texts = list(self._index_struct.embedding_dict.keys())
        fun_rel_texts = self.extract_function_triple(all_rel_texts, function_name='functionality')
        rel_text_embeddings = [
            self._index_struct.embedding_dict[_id] for _id in fun_rel_texts
        ]
        similarities, top_rel_texts = get_top_k_embeddings(
            query_embedding,
            rel_text_embeddings,
            similarity_top_k=self.similarity_top_k,
            embedding_ids=fun_rel_texts,
        )

        # get tool name
        tool_retrieved = []
        for text in top_rel_texts:
            item = re.search(r"(?<=\(')(.*?)(?=\',)", text)
            item = item.group(1)
            tool_retrieved.append(item)
        print(tool_retrieved)

        # get tool subgraph with specific edge type filtering
        tool_subgraph_filter = []

        for tool in tool_retrieved:
            print(tool)
            subgraph = self._graph_store.get_rel_map([tool], 1, limit=self.max_knowledge_sequence)
            count = self.graph_store_query_depth

            extracted_triples1 = []
            for key, triples in subgraph.items():
                for triple in triples:
                    if triple[1] in ['inputs', 'outputs', 'is the input of', 'is the output of',
                                     'has the functionality that']:
                        extracted_triples1.append(triple)

            extracted_triples2 = []
            for triple in extracted_triples1:
                new_sub_graph = self._graph_store.get_rel_map([triple[2]], 1, limit=self.max_knowledge_sequence)
                for key, triples in new_sub_graph.items():
                    for triple in triples:
                        if triple[1] in ['inputs', 'outputs', 'is the input of', 'is the output of',
                                         'has the functionality that']:
                            extracted_triples2.append(triple)

            extracted_triples3 = []
            for triple in extracted_triples2:
                new_sub_graph = self._graph_store.get_rel_map([triple[2]], 1, limit=self.max_knowledge_sequence)
                for key, triples in new_sub_graph.items():
                    for triple in triples:
                        if triple[1] in ['inputs', 'outputs', 'is the input of', 'is the output of',
                                         'has the functionality that']:
                            extracted_triples3.append(triple)

            triples = extracted_triples1 + extracted_triples2 + extracted_triples3
            # remove duplicates from triples
            triples = list(set([tuple(t) for t in triples]))

            start_tool = tool

            # extract tool name nodes
            tool_nodes = set()
            for triple in triples:
                if triple[1] == 'has the functionality that':
                    tool_nodes.add(triple[0])

            # initialize path dictionary
            tool_paths = {}

            # use queue for breadth-first search (BFS)
            from collections import deque

            for tool_node in tool_nodes:
                if tool_node != start_tool:
                    queue = deque([(tool_node, [])])
                    visited = set()
                    found = False
                    while queue and not found:
                        current_node, path = queue.popleft()
                        if current_node in visited:
                            continue
                        visited.add(current_node)
                        for triple in triples:
                            if triple[2] == current_node:
                                new_path = path + [triple]
                                if triple[0] == start_tool:
                                    tool_paths[tool_node] = new_path
                                    found = True
                                    break
                                queue.append((triple[0], new_path))

            # classify paths
            matching_paths = {}
            non_matching_paths = {}

            for tool, path in tool_paths.items():
                contains_pattern = False
                for i in range(len(path) - 1):
                    if (path[i][1] == 'is the input of' and path[i + 1][1] == 'inputs') or (
                            path[i][1] == 'outputs' and path[i + 1][1] == 'is the output of'):
                        contains_pattern = True
                        break

                if not contains_pattern:
                    non_matching_paths[tool] = path

            # save to variable
            # tool_paths_with_pattern = matching_paths
            tool_paths_without_pattern = non_matching_paths

            for tool, path in tool_paths_without_pattern.items():
                # print(f"Tool Node: {tool}")
                tool_subgraph_filter.append(tool)


        # get tool subgraph
        tool_subgraph = self._graph_store.get_rel_map(
            tool_retrieved, self.graph_store_query_depth, limit=self.max_knowledge_sequence
        )

        # compute embedding similarity between subgraph tools with query+previous_tool
        c = 0
        for tool in tool_subgraph:

            fun_rel_texts = self.extract_function_triple(self.list2string(tool_subgraph[tool]),
                                                         function_name='functionality')

            fun_rel_texts = list(dict.fromkeys(fun_rel_texts))  # remove duplicate

            tool_existed = "There is an existing tool:" + str(fun_rel_texts[0]).replace("\'", "").replace("(",
                                                                                                          "").replace(
                ")", "")
            fun_rel_texts = fun_rel_texts[1:]
            if fun_rel_texts == []:
                continue

            query_embedding = self._embed_model.get_text_embedding(query_bundle.query_str)

            fun_rel_texts = [tool for tool in fun_rel_texts if
                             re.search(r"(?<=\(')(.*?)(?=\',)", tool).group(1) in tool_subgraph_filter]

            fun_rel_texts = [tool + tool_existed for tool in fun_rel_texts]


            rel_text_embeddings = self.batch_get_text_embeddings(fun_rel_texts, max_workers=500)

            similarities_sub, top_rel_texts_sub = get_top_k_embeddings(
                query_embedding,
                rel_text_embeddings,
                similarity_top_k=self.similarity_top_k,  # 3
                embedding_ids=fun_rel_texts,
            )
            for text in top_rel_texts_sub:
                item = re.search(r"(?<=\(')(.*?)(?=\',)", text)
                item = item.group(1)
                tool_retrieved.append(item)
            similarities = similarities + [i * similarities[c] for i in similarities_sub]
            c = c + 1

        # remove duplicates
        tool_retrieved_dep = list(dict.fromkeys(tool_retrieved))

        # get tool weights based on similarity
        tool_weights = {}
        for item in tool_retrieved_dep:
            weight = [similarities[i] for i, x in enumerate(tool_retrieved) if x == item]
            tool_weights[item] = sum(weight)

        # Priority is given to larger weights
        tool_retrieved_sort = sorted(tool_retrieved_dep, key=lambda x: tool_weights[x], reverse=True)
        tool_retrieved_sort = tool_retrieved_sort[:self.max_tools]
        # get tool node
        tool_subgraph = self._graph_store.get_rel_map(
            tool_retrieved_sort, 1, limit=self.max_knowledge_sequence
        )
        print(tool_retrieved_sort)


        original_triples = []
        for tool in tool_retrieved_sort:
            original_triples.extend(tool_subgraph[tool])




        return original_triples


    def get_tool_subgraph(self, tool_name, depth=2, limit=1000):
        """
        Retrieve the subgraph for a given tool from the knowledge graph.

        Args:
            tool_name (str): The name of the tool for which to retrieve the subgraph.
            depth (int): The depth of the search in the graph.
            limit (int): The maximum number of nodes to retrieve in the subgraph.

        Returns:
            dict: A dictionary representing the subgraph connected to the tool.
        """
        try:
            subgraph = self._graph_store.get_rel_map([tool_name], depth=depth, limit=limit)
            return subgraph.get(tool_name, {})
        except Exception as e:
            logger.error(f"Failed to retrieve subgraph for tool {tool_name}: {e}")
            return {}
    
    def get_tool_info(self, subgraph: List[List[str]]) -> Dict[str, Any]:
        """
        Extract the tool's input format, output format, and security check requirement from the subgraph.

        Args:
            subgraph (List[List[str]]): The subgraph content representing the tool's relationships.

        Returns:
            Dict[str, Any]: A dictionary containing the input format, output format, and security check requirement.
        """
        input_format = None
        output_format = None
        requires_security_check = False

        for triplet in subgraph:
            if triplet[1] == 'inputs':
                input_format = triplet[2]
            elif triplet[1] == 'outputs':
                output_format = triplet[2]
            elif triplet[1] == 'does not need ' and triplet[2] == 'Security Check':
                requires_security_check = False
            elif triplet[1] == 'needs ' and triplet[2] == 'Security Check':
                requires_security_check = True

        # Return the extracted information
        return {
            'input_format': input_format,
            'output_format': output_format,
            'requires_security_check': requires_security_check
        }


