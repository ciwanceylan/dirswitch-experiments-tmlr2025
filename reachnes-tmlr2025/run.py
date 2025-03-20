import argparse
import numpy as np
import pandas as pd
import torch
import torch_sparse as tsp

import reachnes.adj_utils as rn_adjutils
import reachnes.run_reachnes as rn_run


def fix_edge_orientation(edges):
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    return edges


def _post_load(
    df,
    is_weighted,
    as_canonical_undirected,
    add_symmetrical_edges,
    remove_self_loops=True,
):
    if remove_self_loops:
        df = df.loc[df.iloc[:, 0] != df.iloc[:, 1], :]
    edges = df.iloc[:, [0, 1]].to_numpy().astype(np.int64)
    if is_weighted and df.shape[1] > 2:
        weights = df.iloc[:, 2:].to_numpy().astype(np.float64)
        weights = (
            weights.squeeze(axis=1)
            if weights.ndim > 1 and weights.shape[1] == 1
            else weights
        )
    else:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    if as_canonical_undirected:
        edges = np.sort(edges, axis=1)
        df = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
        weights_df = pd.DataFrame(weights)
        df = pd.concat((df, weights_df), axis=1)
        df = df.groupby(["source", "target"], as_index=False).agg("mean")
        edges = df.loc[:, ["source", "target"]].to_numpy().astype(np.int64)
        weights = df.iloc[:, 2:].to_numpy().astype(np.float64)
        weights = (
            weights.squeeze(axis=1)
            if weights.ndim > 1 and weights.shape[1] == 1
            else weights
        )

        if add_symmetrical_edges:
            sym_edges = np.stack((edges[:, 1], edges[:, 0]), axis=1)
            edges = np.concatenate((edges, sym_edges), axis=0)
            weights = np.concatenate((weights, weights), axis=0)

    return edges, weights


def read_graph_from_npz(
    fp, as_undirected: bool, remove_self_loops: bool = True, use_weights: bool = True
):
    with np.load(fp, allow_pickle=False) as data:
        # Get either 'edge_index' or 'edges_index' from data.
        if "edge_index" in data:
            edges = data["edge_index"]
        elif "edges_index" in data:
            edges = data["edges_index"]
        else:
            edges = None
        edges = fix_edge_orientation(edges)
        node_attributes = data["x"] if "x" in data else None
        num_nodes = data["num_nodes"] if "num_nodes" in data else None
        if num_nodes is None and node_attributes is None:
            num_nodes = edges.max() + 1
        elif num_nodes is None:
            num_nodes = node_attributes.shape[0]
        else:
            num_nodes = num_nodes.item()
        weights = data["edge_attr"] if "edge_attr" in data and use_weights else None
        directed = data["directed"] if "directed" in data else False
        try:
            directed = directed.item()
        except AttributeError:
            pass
        directed = not as_undirected and directed

    data = pd.DataFrame({"source": edges[:, 0], "target": edges[:, 1]})
    if weights is not None:
        weights = pd.DataFrame(weights)
        data = pd.concat((data, weights), axis=1)
    edges, weights = _post_load(
        data,
        is_weighted=weights is not None,
        as_canonical_undirected=not directed,
        add_symmetrical_edges=not directed,
        remove_self_loops=remove_self_loops,
    )
    adj = tsp.SparseTensor(
        row=torch.from_numpy(edges[:, 1]),
        col=torch.from_numpy(edges[:, 0]),
        value=torch.from_numpy(weights),
        sparse_sizes=(num_nodes, num_nodes),
    )

    return adj, node_attributes, directed


def load_adj_obj(data_path: str, force_undirected: bool, dtype: torch.dtype):
    adj, node_attributes, directed = read_graph_from_npz(
        data_path, as_undirected=force_undirected, remove_self_loops=True
    )
    adj_obj = rn_adjutils.TorchAdj(
        adj=adj, make_undirected=False, remove_self_loops=False, dtype=dtype
    )
    return adj_obj


def run():
    parser = argparse.ArgumentParser(description="Run Reachnes node embedding models.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to a .npz file containing the fields: "
        "'num_nodes', 'edge(s)_index', 'x', 'edge_attr', and 'directed'.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path where the embeddings and other results will be saved.",
    )
    parser.add_argument(
        "--force-undirected",
        action="store_true",
        help="Force the graph to be undirected. Else use directed if specified in the file.",
    )
    parser.add_argument("--ddp", action="store_true", help="Run using torch DDP.")
    parser.add_argument(
        "--ddp-backend",
        default="nccl",
        help="Which ddp backend to use. Only 'nccl' should be used for now.",
    )
    parser.add_argument(
        "--num-gpus",
        default=None,
        help="Limit the number of GPUs when using ddp, otherwise all available are used.",
    )
    parser = rn_run.ReachnesArguments.fill_parser(parser)
    args = parser.parse_args()
    excluded_args = {
        "data_path",
        "output_path",
        "force_undirected",
        "ddp",
        "ddp_backend",
        "num_gpus",
    }
    args_kwargs = {k: v for k, v in vars(args).items() if k not in excluded_args}
    rn_args = rn_run.ReachnesArguments(**args_kwargs)
    rn_spec = rn_run.ReachnesSpecification.from_rn_args(rn_args)

    adj_obj = load_adj_obj(
        args.data_path, force_undirected=args.force_undirected, dtype=rn_spec.dtype
    )

    if args.ddp:
        embeddings = rn_run.run_ddp(
            adj_obj, rn_spec, backend=args.ddp_backend, num_gpus=args.num_gpus
        )
    else:
        embeddings = rn_run.run_single_node(adj_obj, rn_spec)

    embeddings = embeddings.detach().cpu().numpy()
    np.save(args.output_path, embeddings)


if __name__ == "__main__":
    run()
