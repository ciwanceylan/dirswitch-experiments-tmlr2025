import os
import sys
import json
import time
import numpy as np
import scipy.sparse as sp
import hashlib

from oct2py import octave
import scipy.sparse.linalg as spl

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))
nebtools_dir = os.path.abspath(
    os.path.join(METHOD_DIR, "..", "..", "..", "src", "nebtools")
)
sys.path.append(nebtools_dir)

import argsfromconfig as parsing
import data.core_ as datacore


# def get_radius_bound(adj):
#     out_degrees = np.asarray(adj.sum(axis=0)).ravel().astype(np.int32)
#     in_degrees = np.asarray(adj.sum(axis=0)).ravel().astype(np.int32)
#     max_in = np.max(in_degrees).item()
#     max_out = np.max(out_degrees).item()
#     return min(max_out, max_in)


# def hash_from_degree_distributions(adj):
#     out_degrees = np.asarray(adj.sum(axis=0)).ravel().astype(np.int32)
#     in_degrees = np.asarray(adj.sum(axis=0)).ravel().astype(np.int32)
#     degrees = np.concatenate((out_degrees, in_degrees))
#     hash_code = hashlib.md5(degrees.tobytes()).hexdigest()
#     return hash_code


# def compute_spectral_radius(adj):
#     tols = [1e-8, 1e-6, 1e-4, 1e-3]
#     radius = None
#     for i, tol in enumerate(tols):
#         try:
#             w = spl.eigs(adj, k=2, return_eigenvectors=False, tol=tol)
#             radius = np.abs(w).max()
#             return radius
#         except spl.ArpackNoConvergence as e:
#             if i == len(tols) - 1:
#                 raise
#             else:
#                 continue
#     return radius


# def get_radius(adj):
#     with open(os.path.join(METHOD_DIR, 'radius.json'), 'r') as fp:
#         radii_dict = json.load(fp)
#     hash_code = hash_from_degree_distributions(adj)
#     if hash_code in radii_dict:
#         radius = radii_dict[hash_code]
#     else:
#         radius = spectral_radius_power_iterations(adj, max_iter=10, tol=1e-2)
#         radii_dict[hash_code] = radius
#         with open(os.path.join(METHOD_DIR, 'radius.json'), 'w') as fp:
#             json.dump(radii_dict, fp=fp, indent=2)
#     return radius
#
#
def spectral_radius_power_iterations(adj, max_iter, tol):
    num_nodes = adj.shape[1]
    v_k = (1.0 / num_nodes) * np.ones(shape=(num_nodes,), dtype=np.float32)
    radius = num_nodes

    for _ in range(max_iter):
        v_k1 = adj @ v_k
        radius = np.linalg.norm(v_k1, ord=2)
        v_k1 = v_k1 / radius
        err = np.minimum(
            np.sqrt(np.sum(np.power(v_k1 - v_k, 2))),
            np.sqrt(np.sum(np.power(v_k1 + v_k, 2))),
        )
        v_k = v_k1
        if err <= tol:
            break
    return radius


def get_hope_embeddings(adj, K, beta_multiplier: float):
    env_location = os.environ["CONDA_PREFIX"]
    path_to_needed_functions = [
        os.path.join(env_location, "share/octave/7.3.0/m/sparse"),
        os.path.join(env_location, "share/octave/7.3.0/m/linear-algebra"),
    ]

    for path in path_to_needed_functions:
        octave.addpath(path)
    octave.addpath(f"{METHOD_DIR}/hope")
    # octave.eval("pkg load svds")
    radius = spectral_radius_power_iterations(adj, max_iter=10, tol=1e-2)
    # radius = get_radius_bound(adj)
    beta = beta_multiplier / radius
    embeddings = octave.feval(f"{METHOD_DIR}/hope/embed_main.m", adj, K, beta)
    return embeddings


def compute_embeddings(
    input_file, output_path, as_undirected, weighted, args, metadata_path=None
):
    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
    )

    directed = not as_undirected and directed
    if not weighted:
        weights = np.ones(weights.shape[0], dtype=np.float64)

    adj = datacore.edges2spmat(edges, weights, num_nodes=num_nodes, directed=directed).T

    start = time.time()
    embeddings = get_hope_embeddings(
        adj, K=args.dimensions // 2, beta_multiplier=args.beta_multiplier
    )
    embeddings = embeddings.astype(np.float32)
    duration = time.time() - start
    meta_data = vars(args)
    meta_data["duration"] = duration

    save_results(
        output_path=output_path,
        metadata_path=metadata_path,
        embeddings=embeddings,
        meta_data=meta_data,
    )


def save_results(output_path, metadata_path, embeddings, meta_data):
    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, "w") as fp:
            json.dump(meta_data, fp)


def main():
    name = "HOPE (reference MATLAB)"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "hope")
    parser.description = f"{name}: Reference implementation for Asymmetric Transitivity Preserving Graph Embedding."
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as fp:
            config = json.load(fp)
    else:
        config = vars(args)

    if config["seed"] is not None:
        np.random.seed(config["seed"])

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
