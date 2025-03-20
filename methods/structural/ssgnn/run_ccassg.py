import os
import time
import json
import random
import numpy as np

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))

import nebtools.argsfromconfig as parsing
import nebtools.data.graph as dgraphs

import torch

import nebtools.ssgnns.ss_gnn_training as ssntrain
import nebtools.ssgnns.cca_ssg.trainutils as ccassg


def get_trainer(graph: dgraphs.SimpleGraph, args, use_cpu: bool):
    device = (
        torch.device(f"cuda")
        if torch.cuda.is_available() and not use_cpu
        else torch.device("cpu")
    )

    params = ccassg.Parameters(
        lr=args.lr,
        wd=args.wd,
        lambd=args.lambd,
        n_layers=args.num_layers,
        drop_edge_ratio=args.der,
        drop_feature_ratio=args.dfr,
        hid_dim=args.dimensions,
        out_dim=args.dimensions,
        model_type=args.encoder,
        dir_seqs=args.dir_seqs.split("::"),
        add_degree=args.add_degree,
        add_lcc=args.add_degree,
        standardize=args.standardize,
    )
    model_trainer = ccassg.CCASSGTrainer(graph=graph, params=params, device=device)

    return model_trainer


def compute_embeddings(
    input_file,
    output_path,
    as_undirected,
    weighted,
    node_attributed,
    args,
    metadata_path=None,
):
    graph = dgraphs.SimpleGraph.load(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
        use_weights=weighted,
        with_node_attributes=node_attributed,
    )

    start = time.perf_counter()
    model_trainer = get_trainer(graph=graph, args=args, use_cpu=args.use_cpu)
    model_trainer, loss_history = ssntrain.train_ss_gnn_without_eval(
        model_trainer=model_trainer, num_epochs=args.num_epochs, verbose=False
    )
    with torch.no_grad():
        embeddings = model_trainer.get_embeddings()
    duration = time.perf_counter() - start
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    meta_data = vars(args)
    meta_data["duration"] = duration
    meta_data["start_loss"] = loss_history[0]
    meta_data["end_loss"] = loss_history[-1]
    np.save(output_path, embeddings, allow_pickle=False)
    if metadata_path is not None:
        with open(metadata_path, "w") as fp:
            json.dump(meta_data, fp)


def main():
    name = "ccassg"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "ccassg")
    parser.description = f"{name}: Self-supervised GNN."
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as fp:
            config = json.load(fp)
    else:
        config = vars(args)

    if config["seed"] is not None:
        np.random.seed(config["seed"])
        random.seed(config["seed"])

    if config["weighted"]:
        raise NotImplementedError(f"Weighted graphs not implemented for {name}")

    if config["edge_attributed"]:
        raise NotImplementedError(f"Edge attributes not implemented for {name}")

    if config["dynamic"]:
        raise NotImplementedError(f"Dynamic graphs not supported for {name}")

    stf_cache_dir = f"/tmp/ssgnn_stf_cache/cpu_{args.cpu_workers}/"
    os.makedirs(stf_cache_dir, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = stf_cache_dir

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
