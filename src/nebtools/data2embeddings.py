from typing import Dict
import os
import json
import numpy as np
import pandas as pd

import nebtools.data.graph as dgraphs
import nebtools.algs.preconfigs as embalgsets
import nebtools.algs.utils as algutils

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_INDEX = os.path.abspath(
    os.path.join(FILE_DIR, "..", "..", "data", "data_index.json")
)


def get_embeddings_from_spec(
    graph: dgraphs.SimpleGraph,
    alg_spec,
    gc_mode: algutils.GRAPH_CAST_MODE = "force",
    resources: algutils.ComputeResources = None,
    seed: int = 42,
    timeout: int = 500,
):
    if resources is None:
        resources = algutils.ComputeResources()

    algs = algutils.EmbeddingAlg.specs2algs(
        alg_specs=[alg_spec],
        graph=graph,
        gc_mode=gc_mode,
        only_weighted=False,
        concat_node_attributes=False,
    )
    emb_generator = algutils.generate_embeddings_from_subprocesses(
        graph, algs, resources=resources, tempdir="/tmp", seed=seed, timeout=timeout
    )

    alg, embeddings, alg_outputs = next(emb_generator)
    return embeddings, alg, alg_outputs


def get_embeddings(
    graph: dgraphs.SimpleGraph,
    alg_name: str,
    emb_dim: int,
    alg_kwargs: Dict = None,
    gc_mode: algutils.GRAPH_CAST_MODE = "force",
    resources: algutils.ComputeResources = None,
    seed: int = 42,
    timeout: int = 500,
):
    if resources is None:
        resources = algutils.ComputeResources()

    alg_specs = embalgsets.get_alg_by_name(
        [alg_name], emb_dim=emb_dim, alg_kwargs=alg_kwargs
    )
    algs = algutils.EmbeddingAlg.specs2algs(
        alg_specs=alg_specs,
        graph=graph,
        gc_mode=gc_mode,
        only_weighted=False,
        concat_node_attributes=False,
    )
    emb_generator = algutils.generate_embeddings_from_subprocesses(
        graph, algs, resources=resources, tempdir="/tmp", seed=seed, timeout=timeout
    )

    alg, embeddings, alg_outputs = next(emb_generator)
    return embeddings, alg, alg_outputs


def read_node_labels(dataset):
    with open(DATA_INDEX, "r") as fp:
        dataset_info = json.load(fp)[dataset]
    node_labels_file = os.path.join(
        dataset_info["datapath"], dataset_info["node_labels_file"]
    )
    if dataset == "ppi_labelled":
        node_labels = pd.DataFrame(np.load(node_labels_file))
    else:
        node_labels = pd.read_json(node_labels_file, typ="series")
    return node_labels
