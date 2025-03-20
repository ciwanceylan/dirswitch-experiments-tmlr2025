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

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))
nebtools_dir = os.path.abspath(
    os.path.join(METHOD_DIR, "..", "..", "..", "src", "nebtools")
)
sys.path.append(nebtools_dir)

import argsfromconfig as parsing
import data.core_ as datacore

from run_reachnesx import args2coeffiecients


def create_reachnes_spec(args, num_nodes):
    use_cpu = not torch.cuda.is_available()
    if use_cpu:
        warnings.warn("CUDA is not available, falling back to CPU.")
    available_memory = 32 if use_cpu else torch.cuda.mem_get_info()[1] // (1024**3)
    use_ddp = not use_cpu and torch.cuda.device_count() > 1

    normalization_seq = tuple(args.adj_seq.split("::"))
    coeffs_specs = args2coeffiecients(args)
    batch_size = int(args.batch_size) if args.batch_size != "auto" else args.batch_size

    rn_spec = rn_run.ReachnesSpecification(
        emb_dim=args.dimensions,
        reduction="rsvd" if num_nodes < 2e3 else "sprsvd",
        reduction_args=dict(),
        coeffs=tuple(coeffs_specs),
        order=args.order,
        normalization_seq=normalization_seq,
        filter=None if args.filter == "none" else args.filter,
        filter_args={"dense2sparse": True, "scaling_factor": "num_nodes"},
        use_float64=args.use_float64,
        use_cpu=use_cpu,
        no_melt=False,
        batch_size=batch_size,
        memory_available=available_memory,
    )
    return rn_spec, use_ddp


def _compute_embeddings(adj_obj, rn_spec, use_ddp):
    start = time.time()
    if use_ddp:
        embeddings = rn_run.run_ddp(adj_obj, rn_spec)
    else:
        embeddings = rn_run.run_single_node(adj_obj, rn_spec)
    embeddings = embeddings.detach().cpu().numpy()
    duration = time.time() - start
    return embeddings, duration


def save_results(output_path, metadata_path, embeddings, meta_data):
    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, "w") as fp:
            json.dump(meta_data, fp)


def compute_embeddings(
    input_file, output_path, as_undirected, weighted, args, metadata_path=None
):
    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
    )

    np_dtype = np.float64 if args.use_float64 else np.float32
    directed = not as_undirected and directed
    if not weighted:
        weights = np.ones(weights.shape[0], dtype=np_dtype)

    rn_spec, use_ddp = create_reachnes_spec(args, num_nodes=num_nodes)

    adj = datacore.edges2spmat(edges, weights, num_nodes=num_nodes, directed=directed)
    adj_obj = rn_adjutils.TorchAdj(
        adj=adj, dtype=rn_spec.dtype, remove_self_loops=False
    )

    embeddings, duration = _compute_embeddings(
        adj_obj=adj_obj, rn_spec=rn_spec, use_ddp=use_ddp
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
    name = "Reachnes (RSVD)"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "reachnes_rsvd")
    parser.description = f"{name}: Proximal node embeddings via reachability."
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
