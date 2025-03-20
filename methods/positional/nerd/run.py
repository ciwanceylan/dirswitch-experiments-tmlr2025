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

METHOD_DIR = os.path.dirname(os.path.realpath(__file__))
nebtools_dir = os.path.abspath(
    os.path.join(METHOD_DIR, "..", "..", "..", "src", "nebtools")
)
sys.path.append(nebtools_dir)

import argsfromconfig as parsing
import data.core_ as datacore
import utils as nebutils


@dc.dataclass(frozen=True)
class NERDParams:
    dimensions: int
    walk_size: int
    samples: int


def save_edges_as_adjacency(fp: io.TextIOBase, edges: np.ndarray):
    for src, dst in edges:
        s = f"{src + 1} {dst + 1} 1"
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
            print(result.returncode, result.stdout, result.stderr)
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


def run(data_path, output_path, num_nodes: int, params: NERDParams, args):
    output_path1 = output_path + "hub"
    output_path2 = output_path + "auth"
    command = [
        "./NERD",
        f"-train",
        f"{data_path}",
        "-output1",
        f"{output_path1}",
        "-output2",
        f"{output_path2}",
        "-size",
        f"{params.dimensions // 2}",
        "-walkSize",
        f"{params.walk_size}",
        "-samples",
        f"{params.samples}",
    ]

    outcome, error_out, duration = run_command(
        command, timeout_time=args.timeout, cwd=f"{METHOD_DIR}/nerd/"
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
        for path in [output_path1, output_path2]:
            df = pd.read_csv(
                path, sep="\s+", index_col=0, skiprows=1, header=None
            ).sort_index()
            embs_ = np.zeros((num_nodes, df.shape[1]), dtype=np.float32)
            embs_[df.index - 1, :] = df.to_numpy(dtype=np.float32)
            embeddings.append(embs_)
        embeddings = np.concatenate(embeddings, axis=1)

    finally:
        for path in [output_path1, output_path2]:
            nebutils.silentremove(path)
    return embeddings, meta_data


def compute_embeddings(
    input_file, output_path, as_undirected, weighted, args, metadata_path=None
):
    nerd_params = NERDParams(
        dimensions=args.dimensions, walk_size=args.walk_size, samples=args.samples
    )

    num_nodes, edges, weights, node_attributes, directed = datacore.read_graph_from_npz(
        input_file,
        as_canonical_undirected=as_undirected,
        add_symmetrical_edges=as_undirected,
        remove_self_loops=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True, mode="w") as adj_file:
        save_edges_as_adjacency(adj_file, edges=edges)

        internal_output_path = adj_file.name[:-4] + "_nerdoutput"
        data_path = os.path.abspath(adj_file.name)
        embeddings, meta_data = run(
            data_path,
            internal_output_path,
            num_nodes=num_nodes,
            params=nerd_params,
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
    name = "NERD (c++)"
    parser = parsing.make_parser(f"{METHOD_DIR}/config.yml", "nerd")
    parser.description = f"{name}: node embeddings from alternating random walks"
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
