from typing import Tuple, List, Dict, Literal
import abc
import dataclasses as dc

import torch
import torch.nn as nn
import torch_sparse as tsp
import torch.distributed as tdistr

import torch_sprsvd.sprsvd as tsprsvd

import reachnes.utils as rn_utils


@dc.dataclass(frozen=True)
class PreEmbeddings:
    data: Dict[str, torch.Tensor]
    batch_indices: torch.LongTensor
    shapes: Dict[str, torch.Size]
    store_modes: Dict[str, Literal["cat", "sum"]]

    def data2send(self):
        return [data for key, data in self.data.items()]


class GatheredPreEmbeddings:
    cat_data: Dict[str, List[torch.Tensor]]
    sum_data: Dict[str, torch.Tensor]
    batch_indices: List[torch.LongTensor]

    def __init__(self):
        self.cat_data = dict()
        self.sum_data = dict()
        self.batch_indices = list()

    def store_pre_embeddings(self, pre_embeddings: PreEmbeddings):
        self.batch_indices.append(pre_embeddings.batch_indices)
        for key, value in pre_embeddings.data.items():
            if pre_embeddings.store_modes[key] == "cat":
                self._store_cat_data(key, value)
            elif pre_embeddings.store_modes[key] == "sum":
                self._store_sum_data(key, value)
            else:
                raise ValueError(
                    f"store_mode '{pre_embeddings.store_modes[key]}' not recognized."
                )

    def _store_cat_data(self, key, value):
        if key not in self.cat_data:
            self.cat_data[key] = [value]
        else:
            self.cat_data[key].append(value)

    def _store_sum_data(self, key, value):
        if key not in self.sum_data:
            self.sum_data[key] = value
        else:
            self.sum_data[key] += value

    def merge(self, merge_dim: int) -> PreEmbeddings:
        batch_indices = torch.cat(self.batch_indices).to(dtype=torch.long)
        data = self.sum_data
        shapes = {key: tensor.shape for key, tensor in data.items()}
        store_modes = {key: "sum" for key in data.keys()}
        for key, tensors in self.cat_data.items():
            data[key] = torch.cat(tensors, dim=merge_dim)
            shapes[key] = data[key].shape
            store_modes[key] = "cat"

        return PreEmbeddings(
            data=data,
            batch_indices=batch_indices,
            shapes=shapes,
            store_modes=store_modes,
        )

    @classmethod
    def tdistr_global_gather_pre_embs(
        cls, local_pre_embeddings: PreEmbeddings, world_size: int
    ):
        local_device = local_pre_embeddings.batch_indices.device

        out = cls()

        num_indices = len(local_pre_embeddings.batch_indices)
        list_of_num_indices = [
            torch.zeros(1, dtype=torch.long, device=local_device)
            for _ in range(world_size)
        ]
        tdistr.all_gather(
            tensor_list=list_of_num_indices,
            tensor=torch.tensor(num_indices, device=local_device, dtype=torch.long),
        )
        batch_indices = [
            torch.empty(size, dtype=torch.long, device=local_device)
            for size in list_of_num_indices
        ]
        tdistr.all_gather(
            tensor_list=batch_indices, tensor=local_pre_embeddings.batch_indices
        )

        out.batch_indices = batch_indices

        for name, tensor in local_pre_embeddings.data.items():
            if local_pre_embeddings.store_modes[name] == "cat":
                shape = local_pre_embeddings.shapes[name]
                pre_emb_sizes = [
                    torch.zeros(len(shape), dtype=torch.long, device=local_device)
                    for _ in range(world_size)
                ]
                tdistr.all_gather(
                    tensor_list=pre_emb_sizes,
                    tensor=torch.tensor(shape, device=local_device, dtype=torch.long),
                )
                gathered_pre_embs = [
                    torch.empty(
                        torch.Size(size), dtype=tensor.dtype, device=local_device
                    )
                    for size in pre_emb_sizes
                ]
                tdistr.all_gather(tensor_list=gathered_pre_embs, tensor=tensor)

                out.cat_data[name] = gathered_pre_embs
            elif local_pre_embeddings.store_modes[name] == "sum":
                tdistr.all_reduce(tensor)
                out.sum_data[name] = tensor
            else:
                raise ValueError(
                    f"Unknown store_mode '{local_pre_embeddings.store_modes[name]}'."
                )
        return out


# def tdistr_gather_pre_embeddings_cat(tensor, shape, local_device, world_size):
#     pre_emb_sizes = [torch.zeros(len(shape), dtype=torch.long, device=local_device) for _ in range(world_size)]
#     tdistr.all_gather(tensor_list=pre_emb_sizes,
#                       tensor=torch.tensor(shape, device=local_device, dtype=torch.long))
#     gathered_pre_embs = [torch.empty(torch.Size(size), dtype=tensor.dtype, device=local_device) for size in
#                          pre_emb_sizes]
#     tdistr.all_gather(tensor_list=gathered_pre_embs, tensor=tensor)


class ReductionModel(nn.Module):
    """
    Interface model for reducing reachability to embeddings, with the reachability arriving in batches and the
    reduction taking place on possibly many machines (typically GPUs).

    The ReductionModel is being used as follows
    ```
        reduction_model = ReductionModel()
        gathered_pre_embs = GatheredPreEmbeddings()
        for batch_indices in batches:
            reachability = compute_a_reachability_batch(input, batch_batch_indices)
            # First, the reachability is compressed in an initial stage. If the reachability can be reduced to embeddings
            # independently of other columns, the embeddings may computed here.
            pre_embedding_batch = reduction_model.reachability2pre_embeddings(reachability_batch=reachability,
                                                                              batch_indices=batch_indices)
            # All pre_embeddings are gathered
            gathered_pre_embs.append(pre_embedding_batch)

        # After computing all pre_embeddings, global communication between workers may be needed.
        if reduction_model.requires_global_pre_emb_gather:
            local_pre_embeddings = reduction_model.reduce_gathered_pre_embeddings(gathered_pre_embs)
            # Merge local and global pre-embeddings
            gathered_pre_embs = GatheredPreEmbeddings.tdistr_global_gather_pre_embs(
                local_pre_embeddings, world_size=self.world_size
            )
        # After global communication, all gathered pre embeddings can be merged to a single pre_embedding object
        pre_embeddings = reduction_model.reduce_gathered_pre_embeddings(gathered_pre_embs)

        # Transform pre-embeddings to embeddings
        emb_series = self.reduction_model(pre_embeddings)
    ```


    """

    def __init__(
        self, requires_pre_emb_gather: bool, emb_type: Literal["structural", "proximal"]
    ):
        super().__init__()
        self.requires_global_pre_emb_gather = requires_pre_emb_gather
        self._emb_type = emb_type
        self._num_series = None

    @property
    def num_series(self):
        if self._num_series is None:
            raise ValueError(
                "Must call `set_emb_dim_and_num_series` before accessing 'num_series'"
            )
        return self._num_series

    def init(self, emb_dim: int, num_series: int, num_nodes: int) -> "ReductionModel":
        raise NotImplementedError

    @property
    def is_proximal(self):
        return self._emb_type == "proximal"

    @property
    def is_structural(self):
        return self._emb_type == "structural"

    @abc.abstractmethod
    def memory_factor(self, num_nodes: int):
        """Estimation of the memory needed per computed embedding.
        Uses to estimate a good batch size for GPU computation."""
        raise NotImplementedError

    @abc.abstractmethod
    def reachability2pre_embeddings(
        self,
        reachability_batch: rn_utils.MultiTorchAdjType,
        batch_indices: torch.LongTensor,
    ) -> "PreEmbeddings":
        """Compress a reachability batch to pre_embeddings."""
        raise NotImplementedError

    def reduce_gathered_pre_embeddings(
        self, gathered_pre_embeddings: GatheredPreEmbeddings
    ) -> PreEmbeddings:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, pre_embeddings: PreEmbeddings):
        """Finalize the (pre) embeddings."""
        raise NotImplementedError


class ECFReduction(ReductionModel):
    def __init__(
        self, *, max_ecf_t_val: float = 100.0, use_energy_distance: bool = False
    ):
        super().__init__(requires_pre_emb_gather=False, emb_type="structural")
        self.use_energy_distance = use_energy_distance
        log_max_val = torch.log(torch.tensor(max_ecf_t_val, dtype=torch.float64))

        self.register_buffer("log_max_val", log_max_val)
        self._num_eval_points = None
        self._num_series = None

    @property
    def num_eval_points(self):
        if self._num_eval_points is None:
            raise ValueError("Must call `init` before accessing 'num_eval_points'")
        return self._num_eval_points

    def init(self, *, emb_dim: int, num_series: int, num_nodes: int):
        self._num_series = num_series
        self._num_eval_points = emb_dim // 2
        return self

    def memory_factor(self, num_nodes: int):
        factor = 2.5 * self.num_series * self.num_eval_points
        return factor

    def reachability2pre_embeddings(
        self,
        reachability_batch: rn_utils.MultiTorchAdjType,
        batch_indices: torch.LongTensor,
    ):
        device = batch_indices.device
        max_val = torch.exp(self.log_max_val)
        ecf_t = (
            max_val
            * torch.arange(1, self.num_eval_points + 1, step=1, device=device)
            / self.num_eval_points
        )
        if isinstance(reachability_batch, torch.Tensor):
            ecf_t = ecf_t.to(dtype=reachability_batch.dtype)
            ecfs = ecf_dense(
                reachability_batch,
                ecf_t=ecf_t,
                use_energy_distance=self.use_energy_distance,
            )
        else:
            ecf_t = ecf_t.to(dtype=reachability_batch[0].dtype())
            ecfs = ecf_multi_sparse(
                reachability_batch,
                ecf_t=ecf_t,
                use_energy_distance=self.use_energy_distance,
            )
        pre_embeddings = PreEmbeddings(
            data={"ecfs": ecfs},
            batch_indices=batch_indices,
            shapes={"ecfs": ecfs.shape},
            store_modes={"ecfs": "cat"},
        )
        return pre_embeddings

    def reduce_gathered_pre_embeddings(
        self, gathered_pre_embeddings: GatheredPreEmbeddings
    ) -> PreEmbeddings:
        return gathered_pre_embeddings.merge(merge_dim=1)

    def forward(self, pre_embeddings: PreEmbeddings):
        return pre_embeddings.data["ecfs"]


def ecf_multi_sparse(
    reachability: Tuple[tsp.SparseTensor],
    ecf_t: torch.Tensor,
    use_energy_distance: bool,
):
    ecfs = [
        ecf_sparse(mat, ecf_t, use_energy_distance=use_energy_distance)
        for mat in reachability
    ]
    ecfs = torch.stack(ecfs, dim=0)
    return ecfs


def ecf_sparse(
    reachability: tsp.SparseTensor, ecf_t: torch.Tensor, use_energy_distance: bool
):
    num_total_nodes = reachability.size(0)
    batch_size = reachability.size(1)
    dtype = reachability.dtype()
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    prop_zeros = torch.zeros(batch_size, device=reachability.device(), dtype=dtype)
    nonzero_cols, num_nonzero = torch.unique(
        reachability.storage.col(), return_counts=True
    )
    prop_zeros[nonzero_cols] = (
        num_total_nodes - num_nonzero.to(dtype=dtype)
    ) / num_total_nodes

    tmp: tsp.SparseTensor = reachability.copy().to(cdtype)
    values = tmp.storage.value()
    res = []
    for i, t in enumerate(ecf_t):
        tmp.set_value_(torch.exp(1j * values * t), layout="coo")
        res_ = tmp.sum(dim=0) / num_total_nodes
        real_part = res_.real

        real_part = real_part + prop_zeros
        res.append(real_part)

        im_part = res_.imag
        res.append(im_part)

    ecf_vals = torch.stack(res, dim=0).T
    if use_energy_distance:
        ecf_vals = ecf_vals / torch.repeat_interleave(ecf_t, 2)

    return ecf_vals.to(dtype=dtype)


def ecf_dense(
    reachability: torch.Tensor, ecf_t: torch.Tensor, use_energy_distance: bool
):
    num_total_nodes = reachability.size(-2)
    dtype = reachability.dtype
    res = []
    for i, t in enumerate(ecf_t):
        res_ = torch.exp(1j * reachability * t).sum(dim=-2) / num_total_nodes
        res.append(res_.real)
        res.append(res_.imag)

    ecf_vals = torch.stack(res, dim=1).transpose(-1, -2)
    if use_energy_distance:
        ecf_vals = ecf_vals / torch.repeat_interleave(ecf_t.view(1, 1, -1), 2)

    return ecf_vals.to(dtype=dtype)


class SortedValuesReduction(ReductionModel):
    def __init__(self):
        super().__init__(requires_pre_emb_gather=False, emb_type="structural")
        self._emb_dim = None

    @property
    def emb_dim(self):
        if self._emb_dim is None:
            raise ValueError(
                "Must call `set_emb_dim_and_num_series` before accessing 'emb_dim'"
            )
        return self._emb_dim

    def init(self, *, emb_dim: int, num_series: int, num_nodes: int):
        self._num_series = num_series
        self._emb_dim = emb_dim
        return self

    def memory_factor(self, num_nodes: int):
        factor = 2.5 * self.num_series * self.emb_dim
        return factor

    def reachability2pre_embeddings(
        self,
        reachability_batch: rn_utils.MultiTorchAdjType,
        batch_indices: torch.LongTensor,
    ) -> PreEmbeddings:
        assert isinstance(reachability_batch, torch.Tensor)
        num_series = reachability_batch.shape[0]

        self_reachability = torch.gather(
            reachability_batch, dim=1, index=batch_indices.expand(num_series, 1, -1)
        )
        mod_reachability = torch.scatter(
            reachability_batch,
            dim=1,
            index=batch_indices.expand(num_series, 1, -1),
            value=-1.0,
        )
        topk_reachability, _ = torch.topk(mod_reachability, self.emb_dim - 1, dim=1)
        pre_embeddings_data = torch.cat((self_reachability, topk_reachability), dim=1)
        pre_embeddings_data = pre_embeddings_data.transpose(1, 2)
        pre_embeddings = PreEmbeddings(
            data={"sorted_values": pre_embeddings_data},
            batch_indices=batch_indices,
            shapes={"sorted_values": pre_embeddings_data.shape},
            store_modes={"sorted_values": "cat"},
        )
        return pre_embeddings

    def reduce_gathered_pre_embeddings(
        self, gathered_pre_embeddings: GatheredPreEmbeddings
    ) -> PreEmbeddings:
        return gathered_pre_embeddings.merge(merge_dim=1)

    def forward(self, pre_embeddings: PreEmbeddings):
        return pre_embeddings.data["sorted_values"]


class CentralMomentsReduction(ReductionModel):
    def __init__(self):
        super().__init__(requires_pre_emb_gather=False, emb_type="structural")
        self._num_moments = None

    @property
    def num_moments(self):
        if self._num_moments is None:
            raise ValueError("Must call `init` before accessing 'num_moments'")
        return self._num_moments

    def init(self, *, emb_dim: int, num_series: int, num_nodes: int):
        self._num_series = num_series
        self._num_moments = emb_dim
        return self

    def memory_factor(self, num_nodes: int):
        factor = 2.5 * self.num_series * self.num_moments
        return factor

    def reachability2pre_embeddings(
        self,
        reachability_batch: rn_utils.MultiTorchAdjType,
        batch_indices: torch.LongTensor,
    ) -> PreEmbeddings:
        assert isinstance(reachability_batch, torch.Tensor)
        moments = compute_central_moments_dense(
            reachability_batch, num_moments=self.num_moments
        )
        pre_embeddings = PreEmbeddings(
            data={"moments": moments},
            batch_indices=batch_indices,
            shapes={"moments": moments.shape},
            store_modes={"moments": "cat"},
        )
        return pre_embeddings

    def reduce_gathered_pre_embeddings(
        self, gathered_pre_embeddings: GatheredPreEmbeddings
    ) -> PreEmbeddings:
        return gathered_pre_embeddings.merge(merge_dim=1)

    def forward(self, pre_embeddings: PreEmbeddings):
        return pre_embeddings.data["moments"]


def compute_central_moments_dense(reachability: torch.Tensor, num_moments: int):
    mean = torch.mean(reachability, dim=-2, keepdim=True)
    moments = [mean]
    zero_mean_rb = reachability - mean
    s = zero_mean_rb
    for m in range(num_moments - 1):
        s = s * zero_mean_rb
        moments.append(torch.mean(s, dim=-2, keepdim=True))

    moments = torch.cat(moments, dim=-2)
    return moments.transpose(-1, -2)


class SVDProximalReduction(ReductionModel):
    """WARNING: this reduction assumes that the node indices in each batch are in consecutive order."""

    def __init__(
        self,
        *,
        num_oversampling: int = 8,
        num_rsvd_iter: int = 6,
        include_v: bool = True,
    ):
        super().__init__(requires_pre_emb_gather=True, emb_type="proximal")
        self._k = None
        self.include_v = include_v
        self.num_oversampling = num_oversampling
        self.num_rsvd_iter = num_rsvd_iter
        self._is_dense = False

    @property
    def k(self):
        if self._k is None:
            raise ValueError("Must call `init` before accessing 'k'")
        return self._k

    def init(self, *, emb_dim: int, num_series: int, num_nodes: int):
        self._num_series = num_series
        self._k = emb_dim if not self.include_v else emb_dim // 2
        return self

    def memory_factor(self, num_nodes: int):
        factor = 3 * num_nodes + 1
        return factor

    def reachability2pre_embeddings(
        self,
        reachability_batch: rn_utils.MultiTorchAdjType,
        batch_indices: torch.LongTensor,
    ):
        # assert isinstance(reachability_batch, torch.Tensor)
        if isinstance(reachability_batch, torch.Tensor):
            self._is_dense = True
            reachability_batch = reachability_batch.transpose(-1, -2)
            pre_embeddings = PreEmbeddings(
                data={"reachability_t": reachability_batch},
                batch_indices=batch_indices,
                shapes={"reachability_t": reachability_batch.shape},
                store_modes={"reachability_t": "cat"},
            )
        else:
            self._is_dense = False
            # reachability_batch is Tuple[tsp.SparseTensor]
            data = dict()
            shapes = dict()
            store_modes = dict()
            reachab_batch: tsp.SparseTensor
            for i, reachab_batch in enumerate(reachability_batch):
                # We transpose matrix and update the column indices to reflect the full tensor
                # Here we assume that the batch_indices are in consecutive order so that they can be adjusted based
                # on the first index.
                data[f"{i}_reachability_t_rows"] = (
                    reachab_batch.storage.col() + batch_indices[0]
                )
                data[f"{i}_reachability_t_cols"] = reachab_batch.storage.row()
                data[f"{i}_reachability_t_vals"] = reachab_batch.storage.value()

                shape = reachab_batch.storage.col().shape
                shapes[f"{i}_reachability_t_rows"] = shape
                shapes[f"{i}_reachability_t_cols"] = shape
                shapes[f"{i}_reachability_t_vals"] = shape

                store_modes[f"{i}_reachability_t_rows"] = "cat"
                store_modes[f"{i}_reachability_t_cols"] = "cat"
                store_modes[f"{i}_reachability_t_vals"] = "cat"
            pre_embeddings = PreEmbeddings(
                data=data,
                batch_indices=batch_indices,
                shapes=shapes,
                store_modes=store_modes,
            )

        return pre_embeddings

    def reduce_gathered_pre_embeddings(
        self, gathered_pre_embeddings: GatheredPreEmbeddings
    ) -> PreEmbeddings:
        if self._is_dense:
            return gathered_pre_embeddings.merge(merge_dim=1)
        else:
            return gathered_pre_embeddings.merge(merge_dim=0)

    def forward(self, pre_embeddings: PreEmbeddings) -> torch.Tensor:
        embeddings = []
        if self._is_dense:
            reachability_t = pre_embeddings.data["reachability_t"]
            num_reachability_series = reachability_t.shape[0]
            for s in range(num_reachability_series):
                pre_embedding_s = reachability_t[s]
                embedding_s = full_svd_reduction(
                    pre_embedding_s, emb_dim=self.k, include_v=self.include_v
                )
                embeddings.append(embedding_s)
        else:
            num_reachability_series = len(pre_embeddings.data.keys()) // 3
            num_nodes = len(pre_embeddings.batch_indices)
            for i in range(num_reachability_series):
                sparse_reachability = tsp.SparseTensor(
                    row=pre_embeddings.data[f"{i}_reachability_t_rows"],
                    rowptr=None,
                    col=pre_embeddings.data[f"{i}_reachability_t_cols"],
                    value=pre_embeddings.data[f"{i}_reachability_t_vals"],
                    sparse_sizes=(num_nodes, num_nodes),
                    is_sorted=False,
                    trust_data=False,
                )
                embedding_s = rsvd_reduction(
                    sparse_reachability,
                    k=self.k,
                    include_v=self.include_v,
                    num_oversampling=self.num_oversampling,
                    num_rsvd_iter=self.num_rsvd_iter,
                )
                embeddings.append(embedding_s)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings


def full_svd_reduction(pre_embeddings: torch.Tensor, emb_dim: int, include_v: bool):
    U, singular_values, Vh = torch.linalg.svd(pre_embeddings)
    return svd2embeddings(U, singular_values, Vh, emb_dim=emb_dim, include_v=include_v)


def rsvd_reduction(
    pre_embeddings: tsp.SparseTensor,
    k: int,
    include_v: bool,
    num_oversampling: int,
    num_rsvd_iter: int,
):
    U, singular_values, Vh = tsprsvd.multi_pass_rsvd(
        pre_embeddings, k=k, num_oversampling=num_oversampling, num_iter=num_rsvd_iter
    )
    return svd2embeddings(U, singular_values, Vh, emb_dim=k, include_v=include_v)


def svd2embeddings(
    U: torch.Tensor,
    singular_values: torch.Tensor,
    Vh: torch.Tensor,
    emb_dim: int,
    include_v: bool,
):
    U = U[:, :emb_dim]
    singular_values = singular_values[:emb_dim]

    # Correct sign if any negative
    sv_sign = torch.sign(singular_values)
    U = U * sv_sign.view(1, -1)
    singular_values = sv_sign * singular_values
    embeddings = U * torch.sqrt(singular_values).view(1, -1)

    if include_v:
        V = Vh.t()
        V = V[:, :emb_dim]
        V_sqrt_s = V * torch.sqrt(singular_values).view(1, -1)
        embeddings = torch.cat((embeddings, V_sqrt_s), dim=1)

    return embeddings


def proximal_embedding_series2decomposition(embedding_series: torch.Tensor):
    out_embs, in_embs = torch.chunk(embedding_series, chunks=2, dim=-1)
    return out_embs, in_embs


def fix_num_oversampling_and_block_size(
    k: int, num_oversampling: int, block_size: int, num_nodes: int
):
    num_oversampling = min(num_nodes - k, num_oversampling)
    block_size_residual = (k + num_oversampling) % block_size
    if block_size_residual > 0 and (k + num_oversampling) == num_nodes:
        # In this the oversampling should not be changed.
        block_size = 1
    elif block_size_residual > 0:
        # Try to increase the num_oversampling to match the block size
        num_oversampling = num_oversampling + block_size - block_size_residual
        num_oversampling, block_size = fix_num_oversampling_and_block_size(
            k, num_oversampling, block_size, num_nodes
        )
    return num_oversampling, block_size


class SPRSVDProximalReduction(ReductionModel):
    def __init__(
        self, num_oversampling: int = 8, block_size: int = 4, include_v: bool = True
    ):
        super().__init__(requires_pre_emb_gather=True, emb_type="proximal")
        self._k = None
        self.num_oversampling = num_oversampling
        self.block_size = block_size
        omega = None
        self.register_buffer("omega", omega)
        self.include_v = include_v

    @property
    def k(self):
        if self._k is None:
            raise ValueError("Must call `init` before accessing 'k'")
        return self._k

    def init(self, *, emb_dim: int, num_series: int, num_nodes: int):
        self._num_series = num_series
        self._k = emb_dim if not self.include_v else emb_dim // 2

        self.num_oversampling, self.block_size = fix_num_oversampling_and_block_size(
            k=self.k,
            num_oversampling=self.num_oversampling,
            block_size=self.block_size,
            num_nodes=num_nodes,
        )
        omega = torch.randn(
            num_series,
            num_nodes,
            self.k + self.num_oversampling,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.register_buffer("omega", omega)
        return self

    def memory_factor(self, num_nodes: int):
        factor = 3.5 * self.num_series * (self.k + self.num_oversampling)
        return factor

    def reachability2pre_embeddings(
        self,
        reachability_batch: rn_utils.MultiTorchAdjType,
        batch_indices: torch.LongTensor,
    ) -> PreEmbeddings:
        num_reachability_series = (
            reachability_batch.shape[0]
            if isinstance(reachability_batch, torch.Tensor)
            else len(reachability_batch)
        )
        pre_embedding_data = dict()
        shapes = dict()
        store_modes = {}
        for s in range(num_reachability_series):
            g, h = tsprsvd.core.calc_gh_batch(
                reachability_batch[s].t(), self.omega[s].squeeze(dim=0)
            )
            pre_embedding_data[f"G_{s}"] = g
            pre_embedding_data[f"H_{s}"] = h
            shapes[f"G_{s}"] = g.shape
            shapes[f"H_{s}"] = h.shape
            store_modes[f"G_{s}"] = "cat"
            store_modes[f"H_{s}"] = "sum"

        pre_embeddings = PreEmbeddings(
            data=pre_embedding_data,
            batch_indices=batch_indices,
            shapes=shapes,
            store_modes=store_modes,
        )
        return pre_embeddings

    def reduce_gathered_pre_embeddings(
        self, gathered_pre_embeddings: GatheredPreEmbeddings
    ) -> PreEmbeddings:
        reduced_data = {}
        shapes = {}
        store_modes = {}
        for key, tensor_list in gathered_pre_embeddings.cat_data.items():
            reduced_data[key] = torch.cat(tensor_list, dim=0)
            shapes[key] = reduced_data[key].shape
            store_modes[key] = "cat"

        for key, tensor in gathered_pre_embeddings.sum_data.items():
            reduced_data[key]: torch.Tensor = tensor
            shapes[key] = reduced_data[key].shape
            store_modes[key] = "sum"

        batch_indices = torch.cat(gathered_pre_embeddings.batch_indices).to(
            dtype=torch.long
        )
        pre_embeddings = PreEmbeddings(
            data=reduced_data,
            batch_indices=batch_indices,
            shapes=shapes,
            store_modes=store_modes,
        )
        return pre_embeddings

    def forward(self, pre_embeddings: PreEmbeddings) -> torch.Tensor:
        all_embeddings = []
        for s in range(self.num_series):
            G = pre_embeddings.data[f"G_{s}"]
            H = pre_embeddings.data[f"H_{s}"]

            U, singular_values, Vh = tsprsvd.core.gh_sp_rsvd_block(
                omega_cols=self.omega[s].squeeze(dim=0),
                G=G,
                H=H,
                k=self.k,
                block_size=self.block_size,
            )
            all_embeddings.append(
                svd2embeddings(
                    U, singular_values, Vh, emb_dim=self.k, include_v=self.include_v
                )
            )
        embeddings = torch.stack(all_embeddings, dim=0)
        return embeddings


# def sort_reachability_columns(mat: rn_utils.TorchAdjType):
#     if isinstance(mat, torch.Tensor):
#         out, _ = torch.sort(mat, dim=1, descending=True)
#     else:
#         out = rn_utils.sort_sparse_tensor_columns(mat_sp=mat)
#     return out


# def get_diagonal_and_set_to_negative(mat: rn_utils.TorchAdjType) -> (torch.Tensor, rn_utils.TorchAdjType):
#     if isinstance(mat, torch.Tensor):
#         out = torch.diagonal(mat, dim1=0, dim2=1)
#         new_mat = torch.diagonal_scatter(mat, src=-out, dim1=0, dim2=1)
#     else:
#         out = mat.get_diag()
#         new_mat = mat.set_diag(values=-out)
#     return out, new_mat


def pca_reduction(
    x: torch.Tensor,
    mode: Literal["individual", "joint"],
    desired_dim: int,
    rtol: float = 1e-7,
):
    num_series, num_nodes, x_dim = x.shape
    if mode == "individual":
        red_dim = max(desired_dim // num_series, 1)
        embeddings, _ = pca_compress_tensor(x, num_dims=desired_dim // num_series)
        embeddings = embeddings.transpose(0, 1).reshape(num_nodes, num_series * red_dim)
    elif mode == "joint":
        x = x.transpose(0, 1).reshape(num_nodes, num_series * x_dim)
        embeddings, _ = pca_compress_matrix(x, num_dims=desired_dim, rtol=rtol)
    else:
        raise ValueError(f"Unknown PCA embedding series reduction mode '{mode}'.")
    return embeddings


def pca_compress_matrix(
    x: torch.Tensor, num_dims: int, rtol: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 2
    U, singular_values, Vh = torch.linalg.svd(A=x, full_matrices=False)
    sv_max = singular_values.max()
    sv_threshold = rtol * sv_max
    rank = (singular_values >= sv_threshold).sum().item()
    k = max(min(rank, num_dims), 1)

    out = U[:, :k] * singular_values[:k].view(1, k)
    return out, Vh


def pca_compress_tensor(
    x: torch.Tensor, num_dims: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_series, num_nodes, x_dim = x.shape
    U, singular_values, Vh = torch.linalg.svd(A=x, full_matrices=False)
    num_dims = min(x_dim, num_dims)
    out = U[:, :, :num_dims] * (
        singular_values[:, :num_dims].view(num_series, 1, num_dims)
    )
    return out, Vh
