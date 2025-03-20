import os
import sys
import time
import json


import numpy as np

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))
nebtools_dir = os.path.abspath(
    os.path.join(METHOD_DIR, "..", "..", "..", "src", "nebtools")
)
sys.path.append(nebtools_dir)

import argsfromconfig as parsing
import data.core_ as datacore

from DGGAN.code.dggan import Config, Model


def make_dggan_graph(edges):
    nodes = set()
    nodes_s = set()
    graph = [{}, {}]
    for source_node, target_node in edges:
        source_node = int(source_node)
        target_node = int(target_node)

        nodes.add(source_node)
        nodes.add(target_node)
        nodes_s.add(source_node)

        if source_node not in graph[0]:
            graph[0][source_node] = []
        if target_node not in graph[1]:
            graph[1][target_node] = []

        graph[0][source_node].append(target_node)
        graph[1][target_node].append(source_node)

    return graph, list(nodes), list(nodes_s)


def compute_embeddings(
    input_file, output_path, as_undirected, weighted, args, metadata_path=None
):
    config = Config(
        n_emb=args.dimensions // 2,
        g_batch_size=args.batch_size,
        d_batch_size=args.batch_size,
        lambda_gen=args.lmbda,
        lambda_dis=args.lmbda,
        lr_gen=args.lr,
        lr_dis=args.lr,
    )

    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
    )
    graph, node_list, node_list_s = make_dggan_graph(edges)
    start = time.time()
    dggan_model = Model(
        graph=graph,
        n_node=num_nodes,
        node_list=node_list,
        node_list_s=node_list_s,
        egs=edges.tolist(),
        config=config,
    )
    gen_emb_matrix, disc_emb_matrix = dggan_model.train()
    emb_matrix = np.concatenate((disc_emb_matrix[0], disc_emb_matrix[1]), axis=1)
    duration = time.time() - start

    meta_data = vars(args)
    meta_data["duration"] = duration
    save_results(
        output_path=output_path,
        metadata_path=metadata_path,
        embeddings=emb_matrix,
        meta_data=meta_data,
    )


def save_results(output_path, metadata_path, embeddings, meta_data):
    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, "w") as fp:
            json.dump(meta_data, fp)


def main():
    name = "DGGAN"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "dggan")
    parser.description = f"{name}: Directed adversarial embeddings"
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as fp:
            config = json.load(fp)
    else:
        config = vars(args)

    if config["weighted"]:
        raise NotImplementedError(f"Weighted not implemented for {name}")

    if config["node_attributed"]:
        raise NotImplementedError(f"Node attributes not implemented for {name}")

    if config["edge_attributed"]:
        raise NotImplementedError(f"Edge attributes not implemented for {name}")

    if config["dynamic"]:
        raise NotImplementedError(f"Dynamic graphs not supported for {name}")

    compute_embeddings(
        input_file=config["input_file"],
        output_path=config["output_file"],
        metadata_path=config["metadata"],
        as_undirected=config["undirected"],
        weighted=config["weighted"],
        args=args,
    )


if __name__ == "__main__":
    main()
