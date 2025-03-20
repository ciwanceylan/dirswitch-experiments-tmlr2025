from typing import Union, Optional, List, Sequence, Dict, Literal, Iterable
import typing
import dataclasses as dc
import hashlib
import os.path
import subprocess
import tempfile
import time
import json

import numpy as np

from nebtools.data.graph import SimpleGraph
import structfeatures.features as stf

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
NEB_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))

GRAPH_CAST_MODE = Literal["alg_compatible", "force", "alg_filter"]


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dc.is_dataclass(o):
            return dc.asdict(o)
        return super().default(o)


@dc.dataclass(frozen=True)
class AlgGraphSupport:
    directed: bool
    weighted: bool
    node_attributed: bool = False
    edge_attributed: bool = False
    dynamic: bool = False

    def is_compatible(
        self, graph: SimpleGraph, disregard_node_attributes: bool = False
    ):
        direction = self.directed or graph.is_undirected
        weighted = self.weighted or not graph.is_weighted
        node_attributed = (
            self.node_attributed
            or not graph.is_node_attributed
            or disregard_node_attributes
        )
        return direction and weighted and node_attributed


@dc.dataclass(frozen=True)
class EmbeddingAlgSpec:
    graph_support: AlgGraphSupport
    name: str
    path: str
    env_name: str
    cpu_workers: Optional[int] = None
    cpu_memory: Optional[int] = None
    gpu_workers: Optional[int] = None
    gpu_memory: Optional[int] = None

    def to_json(self):
        return json.dumps(self, cls=EnhancedJSONEncoder)

    def save_as_json(self, path):
        with open(path, "w") as fp:
            json.dump(self, fp)

    def alg_hash(self) -> str:
        return hashlib.sha224((self.to_json().encode("utf-8"))).hexdigest()


@dc.dataclass(frozen=True)
class GraphCastSpec:
    as_undirected: bool
    as_weighted: bool
    with_node_attributes: bool
    with_edge_attributes: bool = False
    as_dynamic: bool = False

    def __str__(self):
        return (
            f"{'-' if self.as_undirected else 'd'}"
            f"{'w' if self.as_weighted else '-'}"
            f"{'n' if self.with_node_attributes else '-'}"
            f"{'e' if self.with_edge_attributes else '-'}"
            f"{'d' if self.as_dynamic else '-'}"
        )

    @classmethod
    def make_as_graph(cls, graph: SimpleGraph):
        """Produce specifications which reflect the graph."""
        as_undirected = graph.is_undirected
        as_weighted = graph.is_weighted
        with_node_attributes = graph.is_node_attributed
        return cls(
            as_undirected=as_undirected,
            as_weighted=as_weighted,
            with_node_attributes=with_node_attributes,
        )

    def make_compatible(self, alg_spec: EmbeddingAlgSpec, graph: SimpleGraph):
        """Produce cast specifications which are compatible with the algorithm. E.g. if the algorithm does not support
        edge directions, remove them."""
        as_undirected = (
            self.as_undirected
            or not alg_spec.graph_support.directed
            or graph.is_undirected
        )
        as_weighted = (
            self.as_weighted and alg_spec.graph_support.weighted and graph.is_weighted
        )
        with_node_attributes = (
            self.with_node_attributes
            and alg_spec.graph_support.node_attributed
            and graph.is_node_attributed
        )
        return GraphCastSpec(
            as_undirected=as_undirected,
            as_weighted=as_weighted,
            with_node_attributes=with_node_attributes,
        )

    def make_force_spec(self, graph: SimpleGraph):
        """Use this to enforce desired specifications for the graph. If the emb alg does not support the specs,
        it should produce an error."""
        as_undirected = self.as_undirected or graph.is_undirected
        as_weighted = self.as_weighted and graph.is_weighted
        with_node_attributes = self.with_node_attributes and graph.is_node_attributed
        return GraphCastSpec(
            as_undirected=as_undirected,
            as_weighted=as_weighted,
            with_node_attributes=with_node_attributes,
        )


@dc.dataclass(frozen=True)
class EmbeddingAlg:
    spec: EmbeddingAlgSpec
    gc_spec: GraphCastSpec
    only_weighted: bool = False
    concat_node_attributes: bool = False

    @property
    def name(self):
        return f"{self.spec.name}::{str(self.gc_spec)}::{'w' if self.only_weighted else '-'}{'n' if self.concat_node_attributes else '-'}"

    def to_json(self):
        return json.dumps(self, cls=EnhancedJSONEncoder)

    def save_as_json(self, path):
        with open(path, "w") as fp:
            json.dump(self, fp)

    def alg_hash(self) -> str:
        return hashlib.sha224((self.to_json().encode("utf-8"))).hexdigest()

    @staticmethod
    def specs2algs(
        alg_specs: Iterable[EmbeddingAlgSpec],
        graph: SimpleGraph,
        gc_mode: GRAPH_CAST_MODE,
        *,
        only_weighted: bool,
        concat_node_attributes: bool,
    ):
        if only_weighted and not graph.is_weighted:
            raise ValueError("Cannot request 'only_weighted' mode for unweighted graph")
        if concat_node_attributes and not graph.is_node_attributed:
            raise ValueError(
                "Cannot request 'concat_node_attributes' mode for graph without node attributes."
            )
        graph_gc_spec = GraphCastSpec.make_as_graph(graph)
        algs = []
        for alg_spec in alg_specs:
            if gc_mode == "alg_compatible":
                gc_spec = graph_gc_spec.make_compatible(alg_spec, graph)
                algs.append(
                    EmbeddingAlg(
                        gc_spec=gc_spec,
                        spec=alg_spec,
                        only_weighted=only_weighted,
                        concat_node_attributes=concat_node_attributes,
                    )
                )
            elif gc_mode == "force":
                gc_spec = graph_gc_spec.make_force_spec(graph)
                algs.append(
                    EmbeddingAlg(
                        gc_spec=gc_spec,
                        spec=alg_spec,
                        only_weighted=only_weighted,
                        concat_node_attributes=concat_node_attributes,
                    )
                )
            elif gc_mode == "alg_filter":
                if alg_spec.graph_support.is_compatible(
                    graph, disregard_node_attributes=concat_node_attributes
                ):
                    gc_spec = graph_gc_spec.make_compatible(alg_spec, graph)
                    algs.append(
                        EmbeddingAlg(
                            gc_spec=gc_spec,
                            spec=alg_spec,
                            only_weighted=only_weighted,
                            concat_node_attributes=concat_node_attributes,
                        )
                    )
            else:
                raise NotImplementedError(f"Graph case mode {gc_mode} not implemented.")
        return algs


@dc.dataclass(frozen=True)
class EmbAlgOutputs:
    name: str
    alg_hash: str
    emb_dim: int
    subprocess_duration: float
    outcome: str
    error_out: List[str]
    command: str
    metadata: Dict
    feature_descriptions: List[str]


@dc.dataclass(frozen=True)
class ComputeResources:
    cpu_workers: int = 4
    cpu_memory: int = 8
    gpu_workers: int = 0
    gpu_memory: int = 8


def build_python_cmd(
    alg: EmbeddingAlg,
    input_file: str,
    embedding_file: str,
    metadata_file: str,
    resources: ComputeResources,
):
    cmd = f"python {os.path.join(NEB_DIR, alg.spec.path)} {input_file} {embedding_file} --metadata {metadata_file}"

    compute_spec = (
        f" --cpu-workers {resources.cpu_workers if alg.spec.cpu_workers is None else alg.spec.cpu_workers}"
        f" --cpu-memory {resources.cpu_memory if alg.spec.cpu_memory is None else alg.spec.cpu_memory}"
        f" --gpu-workers {resources.gpu_workers if alg.spec.gpu_workers is None else alg.spec.gpu_workers}"
        f" --gpu-memory {resources.gpu_memory if alg.spec.gpu_memory is None else alg.spec.gpu_memory}"
    )
    cmd += compute_spec

    graph_spec = (
        f" --undirected {int(alg.gc_spec.as_undirected)}"
        f" --weighted {int(alg.gc_spec.as_weighted)}"
        f" --node_attributed {int(alg.gc_spec.with_node_attributes)}"
        f" --edge_attributed {int(alg.gc_spec.with_edge_attributes)}"
        f" --dynamic {int(alg.gc_spec.as_dynamic)}"
    )
    cmd += graph_spec

    excluded_fields = set(field.name for field in dc.fields(EmbeddingAlgSpec))
    for k, v in dc.asdict(alg.spec).items():
        if k in excluded_fields:
            continue
        if isinstance(v, bool):
            cmd += f" --{k} {int(v)}"
        else:
            cmd += f" --{k} {v}"
    return cmd


def conda_python_command(
    alg: EmbeddingAlg,
    input_file_name: str,
    output_file_name: str,
    metadata_file_name: str,
    resources: ComputeResources,
):
    conda_command = f"conda run -n {alg.spec.env_name}".split()
    python_command = build_python_cmd(
        alg, input_file_name, output_file_name, metadata_file_name, resources=resources
    ).split()
    command = conda_command + python_command
    return command


def run_command(command, timeout_time: float, cwd: str = None):
    error_out = [""]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            universal_newlines=True,
            timeout=timeout_time,
            cwd=cwd,
        )
        if result.returncode != 0:
            outcome = "fail"
            # print(result.returncode, result.stdout, result.stderr)
            if result.stderr:
                error_out = result.stderr.strip().split("\n")[-10:]
                extensionsToCheck = {
                    "memoryerror",
                    "out of memory",
                    "killed",
                    "CUBLAS_STATUS_NOT_INITIALIZED".lower(),
                }
                for msg in error_out[::-1]:
                    if any(ext in msg.lower() for ext in extensionsToCheck):
                        outcome = "oom"
                        break
        else:
            outcome = "completed"
    except subprocess.TimeoutExpired:
        # print("timed out")
        outcome = "timeout"
    return outcome, error_out


def dispatch_embedding_job(
    input_file_name: str,
    alg: EmbeddingAlg,
    tempdir: str,
    resources: ComputeResources,
    seed: int,
    timeout: float = 2000,
):
    os.makedirs(tempdir, exist_ok=True)
    with (
        tempfile.NamedTemporaryFile(
            dir=tempdir, delete=True, suffix=".npy"
        ) as output_file,
        tempfile.NamedTemporaryFile(
            dir=tempdir, delete=True, suffix=".json"
        ) as metadata_file,
    ):
        command = conda_python_command(
            alg,
            input_file_name,
            output_file.name,
            metadata_file.name,
            resources=resources,
        )
        command += ["--timeout", f"{timeout + 50}", "--seed", f"{seed}"]
        start = time.time()
        outcome, error_out = run_command(command, timeout_time=timeout)
        subprocess_duration = time.time() - start
        # output_file.seek(0)
        # metadata_file.seek(0)
        if outcome == "completed" and os.path.getsize(output_file.name) > 0:
            embeddings = np.load(output_file)
        else:
            embeddings = None

        metadata = None
        feature_descriptions = []
        if outcome == "completed" and os.path.getsize(metadata_file.name) > 0:
            metadata = json.load(metadata_file)
            if "feature_descriptions" in metadata:
                feature_descriptions = metadata["feature_descriptions"]
                del metadata["feature_descriptions"]

        emb_dim = embeddings.shape[1] if embeddings is not None else None
    return (
        embeddings,
        EmbAlgOutputs(
            name=alg.name,
            alg_hash=alg.alg_hash(),
            emb_dim=emb_dim,
            subprocess_duration=subprocess_duration,
            outcome=outcome,
            error_out=error_out,
            command=" ".join(command),
            metadata=metadata,
            feature_descriptions=feature_descriptions,
        ),
    )


def _compute_embeddings(
    graph: SimpleGraph,
    alg: EmbeddingAlg,
    input_file: typing.IO,
    tempdir: str,
    num_expected_nodes: int,
    resources: ComputeResources,
    seed: int,
    timeout: float = 2000,
):
    embeddings, alg_outputs = dispatch_embedding_job(
        input_file.name,
        alg=alg,
        tempdir=tempdir,
        resources=resources,
        seed=seed,
        timeout=timeout,
    )

    alg_outputs = validate_embeddings(embeddings, alg_outputs, num_expected_nodes)

    embeddings, alg_outputs = post_process_embeddings(
        embeddings=embeddings,
        alg=alg,
        alg_outputs=alg_outputs,
        graph=graph,
        input_file=input_file,
        tempdir=tempdir,
        resources=resources,
        seed=seed,
        timeout=timeout,
    )
    return embeddings, alg_outputs


def get_features(graph: SimpleGraph, add_degree: bool, add_lcc: bool):
    features = []
    if add_degree:
        deg_features, feature_names = stf.degree_features(
            edge_index=graph.edges.T,
            num_nodes=graph.num_nodes,
            as_undirected=not graph.directed,
            weights=None,
            dtype=np.float32,
        )
        features.append(deg_features)

    if add_lcc:
        lcc_features, lcc_feature_names = stf.local_clustering_coefficients_features(
            edge_index=graph.edges.T,
            num_nodes=graph.num_nodes,
            as_undirected=not graph.directed,
            weights=None,
            dtype=np.float32,
        )
        features.append(lcc_features)

    if graph.is_node_attributed:
        features.append(graph.node_attributes)

    features = np.concatenate(features, axis=1)
    return features


def generate_embeddings_from_subprocesses(
    graph: SimpleGraph,
    algs: Sequence[EmbeddingAlg],
    tempdir: str,
    resources: ComputeResources,
    seed: int,
    timeout: float = 2000,
):
    num_expected_nodes = graph.num_nodes
    with tempfile.NamedTemporaryFile(
        dir=tempdir, delete=True, suffix=".npz"
    ) as input_file:
        graph.save_npz(input_file)
        input_file.seek(0)
        for alg in algs:
            if alg.spec.name == "node_attribute_only":
                embeddings = get_features(
                    graph,
                    add_degree=not graph.is_node_attributed,
                    add_lcc=not graph.is_node_attributed,
                )
                alg_outputs = EmbAlgOutputs(
                    name=alg.name,
                    alg_hash=alg.alg_hash(),
                    emb_dim=embeddings.shape[1],
                    subprocess_duration=0,
                    outcome="completed",
                    error_out=[""],
                    command="",
                    metadata=dict(),
                    feature_descriptions=[""],
                )
            elif alg.spec.name == "node_attributes_and_structural":
                embeddings = get_features(graph, add_degree=True, add_lcc=True)
                alg_outputs = EmbAlgOutputs(
                    name=alg.name,
                    alg_hash=alg.alg_hash(),
                    emb_dim=embeddings.shape[1],
                    subprocess_duration=0,
                    outcome="completed",
                    error_out=[""],
                    command="",
                    metadata=dict(),
                    feature_descriptions=[""],
                )
            else:
                embeddings, alg_outputs = _compute_embeddings(
                    graph=graph,
                    alg=alg,
                    input_file=input_file,
                    tempdir=tempdir,
                    num_expected_nodes=num_expected_nodes,
                    resources=resources,
                    seed=seed,
                    timeout=timeout,
                )
            yield alg, embeddings, alg_outputs


def validate_embeddings(embeddings, alg_outputs, num_expected_nodes: int):
    if (embeddings is None) and alg_outputs.outcome == "completed":
        alg_outputs = dc.replace(alg_outputs, outcome="fail::unknown")
    elif alg_outputs.outcome == "completed" and (embeddings.shape[1] == 0):
        alg_outputs = dc.replace(
            alg_outputs, outcome="fail::zero_dimensional_embeddings"
        )
    elif (
        alg_outputs.outcome == "completed" and embeddings.shape[0] != num_expected_nodes
    ):
        alg_outputs = dc.replace(
            alg_outputs,
            outcome="missing embeddings",
            error_out=[
                f"Produced {embeddings.shape[0]} but {num_expected_nodes} were needed."
            ],
        )
    return alg_outputs


def post_process_embeddings(
    embeddings: np.ndarray,
    alg: EmbeddingAlg,
    alg_outputs: EmbAlgOutputs,
    graph: SimpleGraph,
    input_file,
    tempdir,
    resources: ComputeResources,
    seed: int,
    timeout: float,
):
    # if alg_outputs.outcome == "completed" and alg.gc_spec.as_weighted and not alg.only_weighted:
    #     embeddings, alg_outputs = _add_unweighted_embeddings(embeddings, alg, alg_outputs, input_file,
    #                                                          tempdir, resources, seed, timeout)

    if (
        alg_outputs.outcome == "completed"
        and alg.concat_node_attributes
        and not alg.gc_spec.with_node_attributes
        and graph.is_node_attributed
    ):
        embeddings = np.concatenate((embeddings, graph.node_attributes), axis=1)
        if alg_outputs.feature_descriptions:
            na_names = [f"na{i}" for i in range(graph.node_attributes.shape[1])]
            feature_descriptions = alg_outputs.feature_descriptions + na_names
            alg_outputs = dc.replace(
                alg_outputs, feature_descriptions=feature_descriptions
            )

    return embeddings, alg_outputs


# def _add_unweighted_embeddings(embeddings, alg, alg_outputs, input_file, tempdir, resources, seed, timeout):
#     new_gc_spec = dc.replace(alg.gc_spec, as_weighted=False)
#     mod_alg = dc.replace(alg, gc_spec=new_gc_spec)
#     unweighted_embeddings, alg_unweighted_outputs = dispatch_embedding_job(input_file.name, alg=mod_alg,
#                                                                            tempdir=tempdir,
#                                                                            resources=resources,
#                                                                            seed=seed, timeout=timeout)
#     if alg_unweighted_outputs.outcome == "completed":
#         embeddings = np.concatenate((embeddings, unweighted_embeddings), axis=1)
#         if alg_outputs.feature_descriptions and alg_unweighted_outputs.feature_descriptions:
#             feature_descriptions = alg_outputs.feature_descriptions + alg_unweighted_outputs.feature_descriptions
#         else:
#             feature_descriptions = []
#         alg_outputs = EmbAlgOutputs(name=f"{alg_outputs.name}",
#                                     alg_hash=alg_outputs.alg_hash,
#                                     emb_dim=embeddings.shape[1],
#                                     subprocess_duration=(
#                                             alg_outputs.subprocess_duration +
#                                             alg_unweighted_outputs.subprocess_duration
#                                     ),
#                                     outcome="completed",
#                                     error_out=alg_outputs.error_out + alg_unweighted_outputs.error_out,
#                                     command=alg_outputs.command,
#                                     metadata=_fuse_alg_metadata_for_embedding_concatenation(
#                                         alg_outputs.metadata,
#                                         alg_unweighted_outputs.metadata),
#                                     feature_descriptions=feature_descriptions
#                                     )
#     return embeddings, alg_outputs


def _fuse_alg_metadata_for_embedding_concatenation(metadata1: Dict, metadata2: Dict):
    for key, item in metadata2.items():
        if key in metadata1:
            if item != metadata1[key]:
                metadata1[key] = (metadata1[key], item)
        else:
            metadata1[key] = item
    return metadata1
