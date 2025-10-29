from collections import defaultdict
from typing import List

import gradio as gr

from graphgen.bases import BaseLLMWrapper
from graphgen.bases.base_storage import BaseGraphStorage
from graphgen.bases.datatypes import Chunk
from graphgen.models import MMKGBuilder
from graphgen.utils import run_concurrent


async def build_mm_kg(
    llm_client: BaseLLMWrapper,
    kg_instance: BaseGraphStorage,
    chunks: List[Chunk],
    progress_bar: gr.Progress = None,
):
    """
    Build multi-modal KG and merge into kg_instance
    :param llm_client: Synthesizer LLM model to extract entities and relationships
    :param kg_instance
    :param chunks
    :param progress_bar: Gradio progress bar to show the progress of the extraction
    :return:
    """
    mm_builder = MMKGBuilder(llm_client=llm_client)

    results = await run_concurrent(
        mm_builder.extract,
        chunks,
        desc="[2/4] Extracting entities and relationships from multi-modal chunks",
        unit="chunk",
        progress_bar=progress_bar,
    )

    nodes = defaultdict(list)
    edges = defaultdict(list)
    for n, e in results:
        for k, v in n.items():
            nodes[k].extend(v)
        for k, v in e.items():
            edges[tuple(sorted(k))].extend(v)

    await run_concurrent(
        lambda kv: mm_builder.merge_nodes(kv, kg_instance=kg_instance),
        list(nodes.items()),
        desc="Inserting entities into storage",
    )

    await run_concurrent(
        lambda kv: mm_builder.merge_edges(kv, kg_instance=kg_instance),
        list(edges.items()),
        desc="Inserting relationships into storage",
    )

    return kg_instance
