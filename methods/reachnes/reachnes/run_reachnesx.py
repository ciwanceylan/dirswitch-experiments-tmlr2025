from typing import Optional
import os
import sys
import json
import warnings
import time
import random
import numpy as np
import torch

import reachnes.run_reachnes as rn_run
import reachnes.adj_utils as rn_adjutils
import reachnes.coeffs as rn_coeffs
import structfeatures.features as stf

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))
nebtools_dir = os.path.abspath(
    os.path.join(METHOD_DIR, "..", "..", "..", "src", "nebtools")
)
sys.path.append(nebtools_dir)

import argsfromconfig as parsing
import data.core_ as datacore


def get_features(
    num_nodes: int,
    edges: np.ndarray,
    weights: Optional[np.ndarray],
    node_attributes: Optional[np.ndarray],
    directed: bool,
    add_degree: bool,
    add_lcc: bool,
    standardize: bool = False,
):
    features = []
    if add_degree or node_attributes is None:
        deg_features, feature_names = stf.degree_features(
            edge_index=edges.T,
            num_nodes=num_nodes,
            as_undirected=not directed,
            weights=weights,
            dtype=np.float32,
        )
        features.append(deg_features)

    if add_lcc or node_attributes is None:
        lcc_features, lcc_feature_names = stf.local_clustering_coefficients_features(
            edge_index=edges.T,
            num_nodes=num_nodes,
            as_undirected=not directed,
            weights=weights,
            dtype=np.float32,
        )
        features.append(lcc_features)

    if node_attributes is not None:
        features.append(node_attributes)

    features = torch.from_numpy(np.concatenate(features, axis=1))
    if standardize:
        features_std = torch.std(features, dim=0, keepdim=True)
        features_std[features_std < 1e-7] = 1.0
        features = (features - torch.mean(features, dim=0, keepdim=True)) / features_std
    return features


def args2coeffiecients(args):
    names = args.rw_distribution.split("::")
    taus = args.tau.split("::")
    locs = args.loc.split("::")
    coeffs_specs = []
    for name, tau, loc in zip(names, taus, locs):
        r = None
        if name.startswith("nbinom"):
            name, r = name.split("_")
            r = int(r)
        spec = rn_coeffs.CoeffsSpec(name, {"tau": float(tau), "r": r}, int(loc))
        coeffs_specs.append(spec)
    return coeffs_specs


def create_reachnes_spec(args):
    use_cpu = not torch.cuda.is_available()
    if use_cpu:
        warnings.warn("CUDA is not available, falling back to CPU.")
    available_memory = 32 if use_cpu else torch.cuda.mem_get_info()[1] // (1024**3)
    use_ddp = not use_cpu and torch.cuda.device_count() > 1

    normalization_seq = tuple(args.adj_seq.split("::"))
    coeffs_specs = args2coeffiecients(args)

    rn_spec = rn_run.ReachnesSpecification(
        emb_dim=args.dimensions,
        reduction="sorted_values",
        reduction_args=dict(),
        coeffs=tuple(coeffs_specs),
        order=args.order,
        normalization_seq=normalization_seq,
        filter=None if args.filter == "none" else args.filter,
        filter_args=None,
        use_float64=args.use_float64,
        use_cpu=use_cpu,
        no_melt=False,
        memory_available=available_memory,
    )
    return rn_spec, use_ddp


def _compute_embeddings(
    adj_obj, rn_spec: rn_run.ReachnesSpecification, x: torch.Tensor, use_ddp: bool
):
    start = time.time()
    if use_ddp:
        raise NotImplementedError
    else:
        embeddings = rn_run.run_single_node(adj_obj, rn_spec, x=x)
    embeddings = embeddings.detach().cpu().numpy()
    duration = time.time() - start
    return embeddings, duration


def save_results(output_path, metadata_path, embeddings, meta_data):
    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, "w") as fp:
            json.dump(meta_data, fp)


def compute_embeddings(
    input_file,
    output_path,
    as_undirected,
    weighted,
    node_attributed,
    args,
    metadata_path=None,
):
    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
    )

    directed = not as_undirected and directed
    if not weighted:
        weights = None
    if not node_attributed or node_attributes is None:
        node_attributes = None
    else:
        node_attributes = np.atleast_2d(node_attributes)

    rn_spec, use_ddp = create_reachnes_spec(args)

    adj = datacore.edges2spmat(edges, weights, num_nodes=num_nodes, directed=directed)
    adj_obj = rn_adjutils.TorchAdj(
        adj=adj, dtype=rn_spec.dtype, remove_self_loops=False
    )
    x = get_features(
        num_nodes=num_nodes,
        edges=edges,
        weights=weights,
        node_attributes=node_attributes,
        directed=directed,
        add_degree=args.use_degree,
        add_lcc=args.use_lcc,
        standardize=args.standardize_input,
    ).to(rn_spec.dtype)

    embeddings, duration = _compute_embeddings(
        adj_obj=adj_obj, rn_spec=rn_spec, x=x, use_ddp=use_ddp
    )

    meta_data = vars(args)
    meta_data["duration"] = duration

    save_results(
        output_path=output_path,
        metadata_path=metadata_path,
        embeddings=embeddings,
        meta_data=meta_data,
    )


def main():
    name = "Reachnes for embedding from node attributes"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "reachnesx")
    parser.description = f"{name}: Node embeddings via reachability."
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as fp:
            config = json.load(fp)
    else:
        config = vars(args)

    if config["seed"] is not None:
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])

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
        node_attributed=config["node_attributed"],
        args=args,
    )


if __name__ == "__main__":
    main()
