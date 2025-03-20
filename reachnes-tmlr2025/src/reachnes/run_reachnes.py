import copy
from typing import Optional, Tuple, List, Literal, Sequence, Union, Dict
import tempfile
import json
import glob
import contextlib
import os
import shutil
import warnings
import argparse
import math
import dataclasses as dc

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as tdistr
from torch.distributed import init_process_group, destroy_process_group

import reachnes.reachnes as rn_rn
import reachnes.reduction as rn_reduc
import reachnes.adj_utils as rn_adjutils
import reachnes.utils as rn_utils
import reachnes.coeffs as rn_coeffs
import reachnes.ew_filtering as rn_filter
import reachnes.reachnes_random_walks as rn_rw


@dc.dataclass(frozen=True)
class ReachnesArguments:
    reduction: str
    emb_dim: int
    coeffs: str
    order: int
    normalization_seq: str
    filter: str
    filter_args: str
    reduction_args: str
    use_float64: bool
    use_cpu: bool
    no_melt: bool
    batch_size: str
    memory_available: int = 8

    @classmethod
    def loads(cls, json_string: str):
        return cls(**json.loads(json_string))

    def dumps(self):
        return json.dumps(dc.asdict(self))

    @staticmethod
    def fill_parser(parser: argparse.ArgumentParser):
        parser.add_argument(
            "reduction",
            type=str,
            help="Which reduction model to use ['ecf', 'sorted_values', 'sprsvd' or 'svd']",
        )
        parser.add_argument("emb_dim", type=int, help="Embedding dimension to use.")
        parser.add_argument(
            "coeffs",
            type=str,
            help="Coefficients to use represented using a list of lists."
            "The string should containing the coefficients and optional arguments,"
            'e.g. \'(("geometric", {"alpha": 0.4 }), ("poisson", {"tau": 2.0 }))\'',
        )
        parser.add_argument(
            "order",
            type=int,
            help="The order of the reachability polynomial, i.e., the maximum number of steps in"
            "the random walk.",
        )
        parser.add_argument(
            "normalization_seq",
            type=str,
            help='List of normalization order of the random-walk polynomial. E.g., \'["OIS", "O"]\' '
            "means the first embedding series should consist of "
            "'out-normalized', 'in-normalized' and then 'symmetrically-normalized' "
            "for higher orders. Then a second series of embeddings should be only 'out-normalized'.",
        )
        parser.add_argument(
            "--reduction-args",
            type=str,
            default="",
            help="Input arguments to reduction model as a json dict string.",
        )
        parser.add_argument(
            "--filter",
            type=str,
            default="",
            help="Apply an optional filter to the reachability before applying reduction.",
        )
        parser.add_argument(
            "--filter-args",
            type=str,
            default="",
            help="Input arguments to filter model as a json dict string. Set 'dense2sparse' to False"
            "to filter without thresholding to sparse.",
        )
        parser.add_argument(
            "--use-float64",
            action="store_true",
            help="Use double precision, otherwise single precision.",
        )
        parser.add_argument(
            "--use-cpu",
            action="store_true",
            help="Force computation on CPU. Otherwise cuda will be used if available.",
        )
        parser.add_argument(
            "--no-melt",
            action="store_true",
            help="Return the embeddings as a 4D tensor where the different orientations and "
            "coefficients are separated over the first two dimensions. "
            "Otherwise a 2D tensor with the nodes over the first dimension.",
        )
        parser.add_argument(
            "--batch-size",
            type=str,
            default="auto",
            help="Batch size used to compute reachability. Either 'auto' or an integer.",
        )
        parser.add_argument(
            "--memory-available",
            type=int,
            default=8,
            help="Set the available amount of memory (either cuda or cpu). This will determine "
            "how large batch sizes are used.",
        )
        return parser

    def get_norm_seqs(self):
        if self.normalization_seq[0] == "[":  # TODO replace with smth better
            normalization_seq = tuple(json.loads(self.normalization_seq))
        else:
            normalization_seq = (self.normalization_seq,)
        return normalization_seq

    def get_reduction_args(self):
        reduc_args = json.loads(self.reduction_args) if self.reduction_args else dict()
        return reduc_args

    def get_coeffs_specs(self):
        coeff_specs = json_str_to_coeff_specs(self.coeffs)
        return coeff_specs

    def get_filter_args(self):
        if self.filter is None:
            return None
        filter_args = json.loads(self.filter_args) if self.filter_args else dict()
        return filter_args


@dc.dataclass(frozen=True)
class ReachnesSpecification:
    emb_dim: int
    reduction: Literal["ecf", "rsvd", "sprsvd", "sorted_values"]
    reduction_args: Dict[str, Union[bool, int, float, str]]
    coeffs: Tuple[rn_coeffs.CoeffsSpec]
    order: int = 10
    normalization_seq: Tuple[Sequence[rn_adjutils.AdjOrientation]] = ("O",)
    filter: Optional[str] = None
    filter_args: Optional[Dict[str, Union[bool, int, float, str]]] = None
    use_float64: bool = False
    use_cpu: bool = False
    no_melt: bool = False
    batch_size: Union[int, Literal["auto"]] = "auto"
    memory_available: int = 8

    @property
    def num_series(self):
        return len(self.coeffs)

    @property
    def dtype(self):
        return torch.float64 if self.use_float64 else torch.float32

    def to_rn_args_str(self):
        return json.dumps(dc.asdict(self))

    @classmethod
    def from_rn_args(cls, rn_args: ReachnesArguments):
        kwargs = dc.asdict(rn_args)
        kwargs["normalization_seq"] = rn_args.get_norm_seqs()
        kwargs["reduction_args"] = rn_args.get_reduction_args()
        kwargs["coeffs"] = rn_args.get_coeffs_specs()
        kwargs["filter_args"] = rn_args.get_filter_args()
        if kwargs["batch_size"] != "auto":
            kwargs["batch_size"] = int(kwargs["batch_size"])
        return cls(**kwargs)

    def get_rn_params(self, num_nodes: int, nnz: int):
        return rn_rn.RNParams(
            emb_dim=self.emb_dim,
            num_nodes=num_nodes,
            nnz=nnz,
            normalization_seq=self.normalization_seq,
            dtype=self.dtype,
            batch_size=self.batch_size,
            memory_available=self.memory_available,
        )

    def get_reduction_model(self):
        reduction_args = dict() if self.reduction_args is None else self.reduction_args
        if self.reduction.lower() == "ecf":
            reduc_model = rn_reduc.ECFReduction(**reduction_args)
        elif self.reduction.lower() == "sprsvd":
            reduc_model = rn_reduc.SPRSVDProximalReduction(**reduction_args)
        elif self.reduction.lower() == "svd" or self.reduction.lower() == "rsvd":
            reduc_model = rn_reduc.SVDProximalReduction(**reduction_args)
        elif self.reduction.lower() == "sorted_values":
            reduc_model = rn_reduc.SortedValuesReduction()
        else:
            raise NotImplementedError(f"Unknown reduction model '{self.reduction}'")
        return reduc_model

    def get_coeffs_model(self, device: str):
        coeffs_obj = rn_coeffs.RWLCoefficientsModel.from_rwl_distributions(
            rwl_distrs=self.coeffs,
            order=self.order,
            normalize=True,
            dtype=self.dtype,
            device=torch.device(device),
        )
        return coeffs_obj

    def get_filter_model(self, num_nodes: int, nnz: int, device: str):
        if not self.filter:
            return None
        if self.filter_args is None:
            filter_args = dict()
        else:
            filter_args = copy.deepcopy(self.filter_args)

        threshold_model = None
        if filter_args.get("dense2sparse", True):
            threshold_model = rn_filter.ThresholdFilter.create_from_graph_size(
                num_nodes=num_nodes, num_edges=nnz, dense2sparse=True
            )

        if self.filter.lower() == "log":
            scaling_factor_mode = filter_args.get("scaling_factor", "num_nodes")
            if scaling_factor_mode == "num_edges":
                scaling_factor = nnz
            elif scaling_factor_mode == "num_nodes":
                scaling_factor = num_nodes
            elif scaling_factor_mode == "sqrt_m_n":
                scaling_factor = np.sqrt(num_nodes) * np.sqrt(nnz)
            else:
                scaling_factor = float(filter_args["scaling_factor"])
            model = rn_filter.LogFilter(
                scaling_factor=scaling_factor, threshold_filter=threshold_model
            )
        elif self.filter.lower() == "threshold":
            model = rn_filter.ThresholdFilter.create_from_graph_size(
                num_nodes=num_nodes,
                num_edges=nnz,
                dense2sparse=self.filter_args.get("dense2sparse", True),
            )
        elif self.filter.lower() == "betainc":
            model = rn_filter.BetaincFilter(
                dtype=self.dtype, threshold_filter=threshold_model, **self.filter_args
            )
        else:
            raise NotImplementedError(f"Unknown filtering model '{self.filter}'")
        model = model.to(device=torch.device(device), dtype=self.dtype)
        return model


@dc.dataclass(frozen=True)
class ReachnesRWSpecification:
    emb_dim: int
    order: int
    coeffs: Tuple[rn_coeffs.CoeffsSpec]
    use_forward_edges: bool
    use_reverse_edges: bool
    num_epochs: int = 50
    sampled_walk_length: int = 20
    walks_per_node: int = 1
    num_negative_samples: int = 1
    sparse: bool = True
    cpu_workers: int = 8
    batch_size: int = 128
    lr: float = 0.01
    use_float64: bool = False
    use_cpu: bool = False
    as_src_dst_tuple: bool = False
    memory_available: int = 8

    def __post_init__(self):
        assert len(self.coeffs) == 1
        assert self.coeffs[0].loc == 0

    @property
    def dtype(self):
        return torch.float64 if self.use_float64 else torch.float32

    def get_coeffs_model(self, device: str):
        coeffs_obj = rn_coeffs.RWLCoefficientsModel.from_rwl_distributions(
            rwl_distrs=self.coeffs,
            order=self.order,
            normalize=True,
            dtype=self.dtype,
            device=torch.device(device),
        )
        return coeffs_obj

    def to_rn_rw_params(self, num_nodes: int):
        sub_embedding_dim = self.emb_dim // 2
        if self.use_forward_edges and self.use_reverse_edges:
            sub_embedding_dim = sub_embedding_dim // 2

        params = rn_rw.RNRWParams(
            num_nodes=num_nodes,
            sub_embedding_dim=sub_embedding_dim,
            use_forward_edges=self.use_forward_edges,
            use_reverse_edges=self.use_reverse_edges,
            sampled_walk_length=self.sampled_walk_length,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            sparse=self.sparse,
            cpu_workers=self.cpu_workers,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        return params


# def update_reduc_args_from_emb_dim(emb_dim: int, reduction_model: str, reduc_args: Dict,
#                                    normalization_seq: Tuple[str], coeffs: Tuple[rn_coeffs.CoeffsSpec]):
#     # TODO UPDATE EMBEDDING DIM CALCULATIONS
#
#     divide_by_two = reduction_model.lower() == "ecf" or ("svd" in reduction_model and reduc_args.get("include_v", True))
#     num_eval_points = emb_dim
#     if divide_by_two:
#         num_eval_points = num_eval_points // 2
#
#     cn = len(normalization_seq) * len(coeffs)
#     embed_expansion_factor = cn
#     num_eval_points = max(num_eval_points // embed_expansion_factor, 1)
#
#     if reduction_model.lower() == "ecf":
#         if "num_eval_points" in reduc_args and num_eval_points != reduc_args["num_eval_points"]:
#             warnings.warn("'num_eval_points' conflicts with 'emb_dim'. Using 'num_eval_points'.")
#         else:
#             reduc_args["num_eval_points"] = num_eval_points
#
#     elif reduction_model.lower() == "svd" or reduction_model.lower() == "sprsvd":
#         if "k" in reduc_args and num_eval_points != reduc_args["k"]:
#             warnings.warn("'k' conflicts with 'emb_dim'. Using 'k'.")
#         else:
#             reduc_args["k"] = num_eval_points
#     elif reduction_model.lower() == "sorted_values":
#         if "emb_dim" in reduc_args and num_eval_points != reduc_args["emb_dim"]:
#             warnings.warn("Sorted values 'emb_dim' conflicts with total 'emb_dim'. Using 'emb_dim'"
#                           "from reduction arguments.")
#         else:
#             reduc_args["emb_dim"] = num_eval_points
#     else:
#         raise NotImplementedError(f"Unknown reduction model '{reduction_model}'")
#
#     return reduc_args


def json_str_to_coeff_specs(json_str: str) -> List[rn_coeffs.CoeffsSpec]:
    tuple_specs = json.loads(json_str)
    specs = []
    for tuple_spec in tuple_specs:
        if len(tuple_spec) == 2:
            coeff_name, coeff_input = tuple_spec
            loc = 0
        elif len(tuple_spec) == 3:
            coeff_name, coeff_input, loc = tuple_spec
        else:
            raise ValueError(f"Invalid coefficient spec {tuple_spec}")
        spec = rn_coeffs.CoeffsSpec(name=coeff_name, kwargs=coeff_input, loc=loc)
        specs.append(spec)
    return specs


def ddp_setup(rank: int, world_size: int, backend: str = "nccl"):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12369"
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def get_node_chunk(num_network_nodes: int, rank: int, world_size: int):
    chunk_size = int(math.ceil(num_network_nodes // world_size))
    start_node = rank * chunk_size
    end_node = min((rank + 1) * chunk_size, num_network_nodes)
    chunck_indices = torch.arange(start_node, end_node, dtype=torch.long, device=rank)
    return chunck_indices


def run_ddp_node(
    rank: int,
    world_size: int,
    input_file: str,
    output_dir: str,
    backend: str,
    rn_args: str,
):
    ddp_setup(rank, world_size, backend=backend)
    rn_args = ReachnesArguments.loads(rn_args)
    rn_spec = ReachnesSpecification.from_rn_args(rn_args)
    # Load adj from memory
    map_location = {"cpu": f"cuda:{rank}", f"cuda:{0}": f"cuda:{rank}"}
    adj = torch.load(input_file, map_location=map_location)
    adj_obj = rn_adjutils.TorchAdj(
        adj=adj, remove_self_loops=False, dtype=rn_spec.dtype
    )

    # Setup Reachnes model
    rn_params = rn_spec.get_rn_params(num_nodes=adj_obj.num_nodes, nnz=adj_obj.nnz())
    coeffs_obj = rn_spec.get_coeffs_model(f"cuda:{rank}")
    reduc_model = rn_spec.get_reduction_model()
    filter_model = rn_spec.get_filter_model(
        num_nodes=adj_obj.num_nodes, nnz=adj_obj.nnz(), device=f"cuda:{rank}"
    )
    model = rn_rn.ReachnesDDP(
        params=rn_params,
        reduction_model=reduc_model,
        coeffs_obj=coeffs_obj,
        ew_filter=filter_model,
        world_size=world_size,
    )
    model.to(device=torch.device(device=f"cuda:{rank}"), dtype=rn_spec.dtype)

    # Sync the model state across the processes. This includes the omega random matrix
    model_state = model.state_dict()
    for key in model_state.keys():
        tdistr.broadcast(model_state[key], src=0)

    chunck_indices = get_node_chunk(adj_obj.num_nodes, rank, world_size)

    embeddings_batch, node_indices_batch = model(
        adj_obj=adj_obj,
        node_indices=chunck_indices,
        melt_embeddings=not rn_spec.no_melt,
    )
    if not reduc_model.requires_global_pre_emb_gather or rank == 0:
        torch.save(
            {"embeddings": embeddings_batch, "node_indices": node_indices_batch},
            os.path.join(output_dir, f"embeddings_rank_{rank}"),
        )
    # Gather embedding tensors on main process and save to disk?
    destroy_process_group()


def read_and_cat_embeddings(output_dir: str):
    embeddings = []
    node_indices = []
    # for rank in range(world_size):
    for path in glob.glob(f"{output_dir}/embeddings_rank_*"):
        loaded = torch.load(path, map_location=torch.device("cpu"))
        embeddings.append(loaded["embeddings"])
        node_indices.append(loaded["node_indices"])
    emb_cat_dim = 1 if len(embeddings[0].shape) == 3 else 0
    embeddings = torch.cat(embeddings, dim=emb_cat_dim)
    assert len(embeddings.shape) == 3 or len(embeddings.shape) == 2
    node_indices = torch.cat(node_indices)
    sorted_indices, sort_order = torch.sort(node_indices)
    if len(embeddings) == 3:
        embeddings = embeddings[:, sort_order, :]
    else:
        embeddings = embeddings[sort_order, :]
    return embeddings


@contextlib.contextmanager
def make_temp_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


def run_single_node(
    adj_obj: rn_adjutils.TorchAdj,
    rn_spec: ReachnesSpecification,
    x: torch.Tensor = None,
):
    device = "cpu" if rn_spec.use_cpu or not torch.cuda.is_available() else "cuda"
    rn_params = rn_spec.get_rn_params(num_nodes=adj_obj.num_nodes, nnz=adj_obj.nnz())
    coeffs_obj = rn_spec.get_coeffs_model(device)
    filter_model = rn_spec.get_filter_model(
        num_nodes=adj_obj.num_nodes, nnz=adj_obj.nnz(), device=device
    )
    adj_obj = adj_obj.to(device=torch.device(device), dtype=rn_spec.dtype)

    if x is None:
        reduc_model = rn_spec.get_reduction_model()

        model = rn_rn.Reachnes(
            params=rn_params,
            reduction_model=reduc_model,
            coeffs_obj=coeffs_obj,
            ew_filter=filter_model,
        )
        model = model.to(device=torch.device(device), dtype=rn_spec.dtype)
        embeddings, _ = model(adj_obj=adj_obj, melt_embeddings=not rn_spec.no_melt)
    else:
        model = rn_rn.ReachnesNodeAttributes(
            params=rn_params, coeffs_obj=coeffs_obj, ew_filter=filter_model
        )
        model = model.to(device=torch.device(device), dtype=rn_spec.dtype)
        x = x.to(device=torch.device(device), dtype=rn_spec.dtype)
        embeddings = model(adj_obj=adj_obj, melt_embeddings=not rn_spec.no_melt, x=x)

    return embeddings


def run_ddp(
    adj_obj: rn_adjutils.TorchAdj,
    rn_spec: ReachnesSpecification,
    backend: str = "nccl",
    num_gpus: Optional[int] = None,
):
    assert adj_obj.adj_ is not None
    rn_args_json = rn_spec.to_rn_args_str()
    try:
        with (
            tempfile.NamedTemporaryFile(delete=False) as fp_input,
            make_temp_directory() as output_dir,
        ):
            torch.save(adj_obj.adj_, fp_input)
            world_size = torch.cuda.device_count() if num_gpus is None else num_gpus
            mp.spawn(
                run_ddp_node,
                args=(world_size, fp_input.name, output_dir, backend, rn_args_json),
                nprocs=world_size,
            )
            embeddings = read_and_cat_embeddings(output_dir=output_dir)
    finally:
        rn_utils.silentremove(fp_input.name)
    return embeddings


def run_rn_rw(
    edge_index: torch.Tensor, num_nodes: int, rn_spec: ReachnesRWSpecification
):
    device = "cpu" if rn_spec.use_cpu or not torch.cuda.is_available() else "cuda"

    params = rn_spec.to_rn_rw_params(num_nodes=num_nodes)

    coeffs_obj = rn_spec.get_coeffs_model(device=device)

    trainer = rn_rw.RNRWTrainer(
        edge_index=edge_index,
        params=params,
        coeffs_obj=coeffs_obj,
        device=torch.device(device),
    )

    trainer.run_training(rn_spec.num_epochs)

    if rn_spec.as_src_dst_tuple:
        return trainer.get_embeddings()
    else:
        return trainer.get_concat_embeddings()
