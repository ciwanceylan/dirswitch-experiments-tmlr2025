import os
import io
import sys
import time
import dataclasses as dc
import json
import tempfile
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse as sp

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))
nebtools_dir = os.path.abspath(
    os.path.join(METHOD_DIR, "..", "..", "..", "src", "nebtools")
)
sys.path.append(nebtools_dir)

import argsfromconfig as parsing
import data.core_ as datacore
import utils as nebutils


@dc.dataclass(frozen=True)
class APPParams:
    dimensions: int
    jump_factor: float
    num_steps: int
    memory_in_gigs: int


def save_edges_as_adjacency(fp: io.TextIOBase, num_nodes: int, edges: np.ndarray):
    adj = sp.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)
    ).tocsr()
    for node, ptr in enumerate(adj.indptr):
        next_ptr = adj.indptr[node + 1] if node + 1 < num_nodes else len(adj.indices)
        if next_ptr - ptr > 0:
            s = f"{node} "
            for p in range(ptr, next_ptr):
                s += f"{adj.indices[p]} "
            fp.write(s + "\n")


def run_command(command, timeout_time: float, cwd: str = None):
    error_out = ""
    start = time.time()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            universal_newlines=True,
            timeout=timeout_time,
            cwd=cwd,
        )
        duration = time.time() - start
        if result.returncode != 0:
            outcome = "fail"
            # print(result.returncode, result.stdout, result.stderr)
            if result.stderr:
                error_out = result.stderr.strip().split("\n")[-10:]
                extensionsToCheck = {"memoryerror", "out of memory", "killed"}
                for msg in error_out[::-1]:
                    if any(ext in msg.lower() for ext in extensionsToCheck):
                        outcome = "oom"
                        break
        else:
            outcome = "completed"
            # print(duration)
    except subprocess.TimeoutExpired:
        # print("timed out")
        outcome = "timeout"
        duration = timeout_time
    return outcome, error_out, duration


def run(data_path, output_path, num_nodes: int, params: APPParams, args):
    command = [
        "java",
        f"-Xmx{params.memory_in_gigs}g",
        "PPREmbedding",
        f"{data_path}",
        f"{output_path}",
        f"{params.dimensions // 2}",
        f"{params.jump_factor}",
        f"{params.num_steps}",
    ]

    outcome, error_out, duration = run_command(
        command, timeout_time=args.timeout, cwd=f"{METHOD_DIR}/src/"
    )
    meta_data = vars(args)
    meta_data["duration"] = duration
    # meta_data = {
    #     "duration": duration,
    #     "refex_steps": refex_steps - 1,
    #     "max_steps": params.max_iter,
    #     "bin_size": params.bin_size,
    #     "corr_thresh": params.corr_thresh,
    #     "memory_in_megs": params.memory_in_megs
    # }
    if outcome != "completed":
        print(error_out)
    try:
        embeddings = []
        for path in [output_path + "_vec.txt", output_path + "_vec_con.txt"]:
            df = pd.read_csv(path, index_col=0, header=None).sort_index()
            embs_ = np.zeros((num_nodes, df.shape[1] - 1), dtype=np.float32)
            embs_[df.index, :] = df.to_numpy(dtype=np.float32)[
                :, :-1
            ]  # Remove nans in last column
            embeddings.append(embs_)
        embeddings = np.concatenate(embeddings, axis=1)

    finally:
        for path in [output_path + "_vec.txt", output_path + "_vec_con.txt"]:
            nebutils.silentremove(path)
    return embeddings, meta_data


def compute_embeddings(
    input_file, output_path, as_undirected, weighted, args, metadata_path=None
):
    app_params = APPParams(
        dimensions=args.dimensions,
        jump_factor=args.jump_factor,
        num_steps=args.steps,
        memory_in_gigs=args.cpu_memory,
    )

    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True, mode="w") as adj_file:
        save_edges_as_adjacency(adj_file, num_nodes=num_nodes, edges=edges)

        internal_output_path = adj_file.name[:-4] + "_appoutput"
        data_path = os.path.abspath(adj_file.name)
        embeddings, meta_data = run(
            data_path,
            internal_output_path,
            num_nodes=num_nodes,
            params=app_params,
            args=args,
        )

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
    name = "APP (java)"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "app")
    parser.description = f"{name}: directed version of Node2Vec"
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
