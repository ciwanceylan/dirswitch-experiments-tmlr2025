import dataclasses as dc
import typing
from typing import Optional, Literal, Tuple, Sequence, Union
import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse as tsp

from reachnes.utils import AdjType, TorchAdjType

AdjNorm = Literal["none", "out", "in", "sym"]
AdjOrientation = Literal["I", "O", "F", "B", "U", "C", "S", "A", "T", "X"]


class ClearedCacheError(Exception):
    pass


@dc.dataclass(frozen=True)
class AdjSeq:
    seq: Union[str, Sequence[AdjOrientation]]

    def sequence(self, order: int) -> Tuple[AdjOrientation]:
        padding = order - len(self.seq)
        seq = tuple(self.seq + padding * self.seq[-1])
        return seq

    def __post_init__(self):
        assert len(self.seq) > 0
        for s in self.seq:
            assert s in typing.get_args(AdjOrientation)


class TorchAdj:
    adj_: Optional[TorchAdjType]
    is_sparse: bool
    _num_nodes: int
    _cached_device: Optional[torch.device]
    _cached_dtype: Optional[torch.dtype]
    _cached_AD: Optional[TorchAdjType]
    _cached_DA: Optional[TorchAdjType]
    _cached_DAD: Optional[TorchAdjType]
    _cached_UD: Optional[TorchAdjType]
    _cached_DUD: Optional[TorchAdjType]
    cached_adj: str

    def __init__(
        self,
        adj: Optional[AdjType],
        dtype: torch.dtype,
        remove_self_loops: bool,
        cache_norm_adj: bool = False,
        cache_undir_adj: bool = True,
        free_adj_when_caching: bool = False,
        num_nodes: Optional[int] = None,
        _cached_device: Optional[torch.device] = None,
        _cached_dtype: Optional[torch.dtype] = None,
        _cached_AD: Optional[TorchAdjType] = None,
        _cached_DA: Optional[TorchAdjType] = None,
        _cached_DAD: Optional[TorchAdjType] = None,
        _cached_UD: Optional[TorchAdjType] = None,
        _cached_DUD: Optional[TorchAdjType] = None,
    ):
        if adj is None:
            assert num_nodes is not None
            self._num_nodes = num_nodes
            self.adj_ = adj
        else:
            self.adj_: TorchAdjType = to_torch_adj(
                adj, dtype=dtype, remove_self_loops=remove_self_loops
            )

        self._num_nodes = self.adj_.size(0)
        self.is_sparse = isinstance(self.adj_, tsp.SparseTensor)
        self.cache_norm_adj = cache_norm_adj
        self.cache_undir_adj = cache_undir_adj
        self.free_adj_when_caching = free_adj_when_caching
        self._cached_device: Optional[torch.device] = _cached_device
        self._cached_dtype: Optional[torch.dtype] = _cached_dtype
        self._cached_AD: Optional[TorchAdjType] = _cached_AD
        self._cached_DA: Optional[TorchAdjType] = _cached_DA
        self._cached_DAD: Optional[TorchAdjType] = _cached_DAD
        self._cached_UD: Optional[TorchAdjType] = _cached_UD
        self._cached_DUD: Optional[TorchAdjType] = _cached_DUD
        self.cached_adj = ""

    @property
    def num_nodes(self):
        return self._num_nodes

    def nnz(self):
        nnz = self.adj_.nnz() if self.is_sparse else self._num_nodes * self._num_nodes
        return nnz

    # @classmethod
    # def create_with_auto_caching(cls, adj: AdjType, adj_seq: AdjSeq, undirected: bool, dtype: torch.dtype,
    #                              remove_self_loops: bool):
    #     num_unique_orientations = len(set(adj_seq.seq))
    #     free_adj_when_caching = True
    #     cache_norm_adj = num_unique_orientations == 1
    #
    #     return cls(adj=adj, make_undirected=undirected, cache_norm_adj=cache_norm_adj,
    #                free_adj_when_caching=free_adj_when_caching, dtype=dtype, remove_self_loops=remove_self_loops)

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        dtype = dtype if dtype is not None else self.dtype
        adj_ = (
            self.adj_.to(device=device, dtype=dtype) if self.adj_ is not None else None
        )

        _cached_AD = (
            self._cached_AD.to(device=device, dtype=dtype)
            if self._cached_AD is not None
            else None
        )
        _cached_DA = (
            self._cached_DA.to(device=device, dtype=dtype)
            if self._cached_DA is not None
            else None
        )
        _cached_DAD = (
            self._cached_DAD.to(device=device, dtype=dtype)
            if self._cached_DAD is not None
            else None
        )

        _cached_UD = (
            self._cached_UD.to(device=device, dtype=dtype)
            if self._cached_UD is not None
            else None
        )
        _cached_DUD = (
            self._cached_DUD.to(device=device, dtype=dtype)
            if self._cached_DUD is not None
            else None
        )

        new_obj = TorchAdj(
            adj=adj_,
            num_nodes=self.num_nodes,
            dtype=dtype,
            remove_self_loops=False,
            _cached_AD=_cached_AD,
            _cached_DA=_cached_DA,
            _cached_DAD=_cached_DAD,
            _cached_UD=_cached_AD,
            _cached_DUD=_cached_DUD,
            _cached_device=device,
        )
        return new_obj

    def _del_adj(self):
        self._cached_device = self.device
        self._cached_dtype = self.dtype
        del self.adj_
        self.adj_ = None

    @property
    def device(self):
        if self.adj_ is None:
            return self._cached_device
        if self.is_sparse:
            return self.adj_.device()
        else:
            return self.adj_.device

    @property
    def dtype(self):
        if self.adj_ is None:
            return self._cached_dtype
        if self.is_sparse:
            return self.adj_.dtype()
        else:
            return self.adj_.dtype

    def check_cache_error(self):
        if (
            self.adj_ is None
            and (self.cache_norm_adj or self.cache_undir_adj)
            and self.free_adj_when_caching
        ):
            raise ClearedCacheError(
                f"Adjacency matrix has been cleared when caching {self.cached_adj}. "
                f"Turn off caching if multiple normalizations are used."
            )

    @property
    def AD(self):
        if self._cached_AD is not None:
            return self._cached_AD

        self.check_cache_error()

        rw_out_norm_adj = rw_adj_normalization(self.adj_, use_out_degrees=True)

        if self.cache_norm_adj:
            self._cached_AD = rw_out_norm_adj
            if self.free_adj_when_caching:
                self.cached_adj = "AD"
                self._del_adj()
        return rw_out_norm_adj

    @property
    def DA(self):
        if self._cached_DA is not None:
            return self._cached_DA

        self.check_cache_error()

        rw_in_norm_adj = rw_adj_normalization(self.adj_, use_out_degrees=False)

        if self.cache_norm_adj:
            self._cached_DA = rw_in_norm_adj
            if self.free_adj_when_caching:
                self.cached_adj = "DA"
                self._del_adj()
        return rw_in_norm_adj

    @property
    def UD(self):
        if self._cached_UD is not None:
            return self._cached_UD

        self.check_cache_error()

        undir_adj = to_symmetric(self.adj_)
        rw_out_norm_adj = rw_adj_normalization(undir_adj, use_out_degrees=True)

        if self.cache_undir_adj:
            self._cached_UD = rw_out_norm_adj
            if self.free_adj_when_caching:
                self.cached_adj = "UD"
                self._del_adj()
        return rw_out_norm_adj

    @property
    def DAD(self):
        if self._cached_DAD is not None:
            return self._cached_DAD

        self.check_cache_error()

        sym_norm_adj = sym_adj_normalization(self.adj_)

        if self.cache_undir_adj:
            self._cached_DAD = sym_norm_adj
            if self.free_adj_when_caching:
                self.cached_adj = "DAD"
                self._del_adj()
        return sym_norm_adj

    @property
    def DUD(self):
        if self._cached_DUD is not None:
            return self._cached_DUD

        self.check_cache_error()
        undir_adj = to_symmetric(self.adj_)
        sym_norm_adj = sym_adj_normalization(undir_adj)

        if self.cache_norm_adj:
            self._cached_DUD = sym_norm_adj
            if self.free_adj_when_caching:
                self.cached_adj = "DUD"
                self._del_adj()
        return sym_norm_adj


def to_torch_adj(
    adj: AdjType, dtype: torch.dtype, remove_self_loops: bool
) -> TorchAdjType:
    """Convert adjacency matrix to a pytorch format."""
    if isinstance(adj, tsp.SparseTensor) or isinstance(adj, torch.Tensor):
        pass
    elif isinstance(adj, sp.spmatrix) or isinstance(adj, sp.sparray):
        adj = tsp.SparseTensor.from_scipy(adj)
    elif isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    else:
        raise TypeError(f"Unsupported type {type(adj)} for adjacency matrix.")

    # if make_undirected:
    #     adj = to_symmetric(adj)

    if remove_self_loops:
        adj = remove_diag(adj)
    return adj.to(dtype=dtype)


def to_cs_graph(adj: AdjType) -> TorchAdjType:
    """Convert adjacency matrix to a pytorch format."""
    if isinstance(adj, tsp.SparseTensor):
        cs_graph = adj.to_scipy(layout="coo")
    elif isinstance(adj, torch.Tensor):
        cs_graph = sp.csgraph.csgraph_from_dense(adj.numpy())
    elif isinstance(adj, np.ndarray):
        cs_graph = sp.csgraph.csgraph_from_dense(adj)
    elif isinstance(adj, sp.spmatrix) or isinstance(adj, sp.sparray):
        cs_graph = adj
        pass
    else:
        raise TypeError(f"Unsupported type {type(adj)} for adjacency matrix.")

    return cs_graph


def calc_weighted_degrees_dense(adj: torch.Tensor, in_degrees: bool):
    """Calculate the weighted degrees for a dense adjacency matrix."""
    if in_degrees:
        return torch.squeeze(adj.sum(dim=1))
    else:
        return torch.squeeze(adj.sum(dim=0))


def calc_weighted_degrees_tsp(adj: tsp.SparseTensor, in_degrees: bool):
    """Calculate the weighted degrees for a sparse adjacency matrix."""
    if in_degrees:
        return adj.sum(dim=1)
    else:
        return adj.sum(dim=0)


def calc_unweighted_degrees_dense(adj: torch.Tensor, in_degrees: bool):
    """Calculate the degrees for a dense adjacency matrix."""
    nonzero = adj.nonzero()
    if in_degrees:
        num_nodes = adj.shape[0]
        degrees_index, degrees_ = torch.unique(nonzero[:, 0], return_counts=True)
    else:
        num_nodes = adj.shape[1]
        degrees_index, degrees_ = torch.unique(nonzero[:, 1], return_counts=True)
    degrees = torch.zeros(num_nodes, dtype=adj.dtype)
    degrees[degrees_index] = degrees_.to(dtype=adj.dtype)
    return degrees


def calc_unweighted_degrees_tsp(adj: tsp.SparseTensor, in_degrees: bool):
    """Calculate the degrees for a sparse adjacency matrix."""
    if in_degrees:
        num_nodes = adj.size(0)
        degrees_index, degrees_ = torch.unique(adj.storage.row(), return_counts=True)
    else:
        num_nodes = adj.size(1)
        degrees_index, degrees_ = torch.unique(adj.storage.col(), return_counts=True)
    degrees = torch.zeros(num_nodes, dtype=adj.dtype())
    degrees[degrees_index] = degrees_.to(dtype=adj.dtype())
    return degrees


def calc_degrees(adj: TorchAdjType, weighted: bool, in_degrees: bool):
    """Wrapper function for the 4 different degree types."""
    if weighted and isinstance(adj, torch.Tensor):
        degs = calc_weighted_degrees_dense(adj, in_degrees=in_degrees)
    elif not weighted and isinstance(adj, torch.Tensor):
        degs = calc_unweighted_degrees_dense(adj, in_degrees=in_degrees)
    elif weighted and isinstance(adj, tsp.SparseTensor):
        degs = calc_weighted_degrees_tsp(adj, in_degrees=in_degrees)
    elif not weighted and isinstance(adj, tsp.SparseTensor):
        degs = calc_unweighted_degrees_tsp(adj, in_degrees=in_degrees)
    else:
        raise NotImplementedError(
            f"out degrees for weighted={weighted} not implemented for type {type(adj)}"
        )
    return degs


def calc_out_degrees(adj: AdjType, weighted: bool):
    """Alias function for calculating out-degrees using calc_degrees"""
    return calc_degrees(adj, weighted=weighted, in_degrees=False)


def calc_in_degrees(adj: AdjType, weighted: bool):
    """Alias function for calculating in-degrees using calc_degrees"""
    return calc_degrees(adj, weighted=weighted, in_degrees=True)


def calc_avg_branch_factor(adj: sp.spmatrix):
    """Calculate the average branch factor for the graph"""
    out_degrees_ = calc_out_degrees(adj, weighted=False)
    beta = np.mean(out_degrees_)
    return beta


def remove_diag(adj: TorchAdjType):
    """Remove self-loops by setting diagonal to zero."""
    if isinstance(adj, tsp.SparseTensor):
        adj = adj.remove_diag()
    elif isinstance(adj, torch.Tensor):
        adj = torch.diagonal_scatter(
            adj, torch.zeros(adj.shape[0], dtype=adj.dtype, device=adj.device)
        )
    return adj


def add_self_loops_to_zero_deg_nodes(adj: TorchAdjType, degrees: torch.Tensor):
    if isinstance(adj, tsp.SparseTensor):
        zero_deg_nodes = torch.nonzero(~(degrees > 0)).view(-1)
        rows = zero_deg_nodes
        cols = zero_deg_nodes
        values = torch.ones(
            (len(zero_deg_nodes)), dtype=adj.dtype(), device=adj.device()
        )
        new_diag = tsp.SparseTensor(
            row=rows, col=cols, value=values, sparse_sizes=tuple(adj.sizes())
        )
        adj = adj + new_diag
    else:
        eye_values = torch.zeros_like(degrees)
        eye_values[~(degrees > 0)] = 1
        adj = adj + torch.diag(eye_values)
    return adj


def sym_adj_normalization(adj: TorchAdjType) -> TorchAdjType:
    if isinstance(adj, tsp.SparseTensor):
        return sym_adj_normalization_tsp(adj)
    else:
        return sym_adj_normalization_dense(adj)


def sym_adj_normalization_tsp(adj: tsp.SparseTensor) -> tsp.SparseTensor:
    """
    Compute symmetrical normalised adjacency matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.

    Args:
        adj: Weighted adjacency matrix

    Returns:
        adj_norm: Symmetrical normalised Laplacian

    """

    out_degs = torch.sqrt(calc_out_degrees(adj, weighted=True))
    out_degs_inv = torch.zeros_like(out_degs)
    out_degs_inv[out_degs > 0] = torch.reciprocal(out_degs[out_degs > 0])

    in_degs = torch.sqrt(calc_in_degrees(adj, weighted=True))
    in_degs_inv = torch.zeros_like(in_degs)
    in_degs_inv[in_degs > 0] = torch.reciprocal(in_degs[in_degs > 0])

    norm_adj = adj.mul(out_degs_inv.unsqueeze(0))
    norm_adj = norm_adj.mul(in_degs_inv.unsqueeze(-1))

    norm_adj = add_self_loops_to_zero_deg_nodes(norm_adj, degrees=out_degs + in_degs)

    return norm_adj


def sym_adj_normalization_dense(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetrical normalised adjacency matrix for a dense matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.

    Args:
        adj: Weighted adjacency matrix

    Returns:
        adj_norm: Symmetrically normalised adjacency matrix.

    """

    out_degs = torch.sqrt(calc_weighted_degrees_dense(adj, in_degrees=False))
    out_degs_inv = torch.zeros_like(out_degs)
    out_degs_inv[out_degs > 0] = torch.reciprocal(out_degs[out_degs > 0])

    in_degs = torch.sqrt(calc_weighted_degrees_dense(adj, in_degrees=True))
    in_degs_inv = torch.zeros_like(in_degs)
    in_degs_inv[in_degs > 0] = torch.reciprocal(in_degs[in_degs > 0])

    norm_adj = (
        adj.to(dtype=adj.dtype) * out_degs_inv.unsqueeze(0)
    ) * in_degs_inv.unsqueeze(-1)

    norm_adj = add_self_loops_to_zero_deg_nodes(norm_adj, degrees=out_degs + in_degs)
    return norm_adj


def rw_adj_normalization(
    adj: TorchAdjType, use_out_degrees: bool = True
) -> TorchAdjType:
    if isinstance(adj, tsp.SparseTensor):
        return rw_adj_normalization_tsp(adj, use_out_degrees=use_out_degrees)
    else:
        return rw_adj_normalization_dense(adj, use_out_degrees=use_out_degrees)


def rw_adj_normalization_tsp(
    adj: tsp.SparseTensor, use_out_degrees=True
) -> tsp.SparseTensor:
    """
    Compute the random walk normalized adjacency matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.
    Diagonal values equal to 1 are added to nodes without (in) edges.

    Args:
        adj: Weighted adjacency matrix
        use_out_degrees: Normalize using out-degrees. Otherwise, in-degrees.

    Returns:
        adj_norm: Random walk normalized adjacency matrix

    """
    if use_out_degrees:
        degs = out_degs = calc_out_degrees(adj, weighted=True)
        out_degs_inv = torch.zeros_like(out_degs)
        out_degs_inv[out_degs > 0] = torch.reciprocal(out_degs[out_degs > 0])
        norm_adj = adj.mul(out_degs_inv.view(1, -1))

    else:
        degs = in_degs = calc_in_degrees(adj, weighted=True)
        in_degs_inv = torch.zeros_like(in_degs)
        in_degs_inv[in_degs > 0] = torch.reciprocal(in_degs[in_degs > 0])
        norm_adj = adj.mul(in_degs_inv.view(-1, 1))

    norm_adj = add_self_loops_to_zero_deg_nodes(norm_adj, degrees=degs)

    return norm_adj


def rw_adj_normalization_dense(adj: torch.Tensor, use_out_degrees=True) -> torch.Tensor:
    """
    Compute random walk normalised adjacency matrix from dense adjacency matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.
    Diagonal values equal to 1 are added to nodes without (in) edges.

    Args:
        adj: Weighted adjacency matrix
        use_out_degrees: Normalise using out-degrees. Otherwise, in-degrees.

    Returns:
        adj_norm: Random walk normalised adjacency matrix

    """
    if use_out_degrees:
        degs = out_degs = calc_weighted_degrees_dense(adj, in_degrees=False)
        out_degs_inv = torch.zeros_like(out_degs)
        out_degs_inv[out_degs > 0] = torch.reciprocal(out_degs[out_degs > 0])
        norm_adj = adj.to(dtype=adj.dtype) * out_degs_inv.view(1, -1)
    else:
        degs = in_degs = calc_weighted_degrees_dense(adj, in_degrees=True)
        in_degs_inv = torch.zeros_like(in_degs)
        in_degs_inv[in_degs > 0] = torch.reciprocal(in_degs[in_degs > 0])
        norm_adj = adj.to(dtype=adj.dtype) * in_degs_inv.view(-1, 1)

    norm_adj = add_self_loops_to_zero_deg_nodes(norm_adj, degrees=degs)
    return norm_adj


def torch_adj_union(adj1: TorchAdjType, adj2: TorchAdjType):
    assert type(adj1) == type(adj2)
    if isinstance(adj1, torch.Tensor):
        merged_adj = torch.block_diag(adj1, adj2)
    elif isinstance(adj1, tsp.SparseTensor):
        merged_adj = tsp.cat([adj1, adj2], dim=(0, 1))
    else:
        raise ValueError(f"Unsupported merge type '{type(adj1)}")
    return merged_adj


def to_symmetric(adj: AdjType):
    if isinstance(adj, tsp.SparseTensor):
        adj = adj.to_symmetric(reduce="sum")
    elif isinstance(adj, torch.Tensor):
        adj = adj + adj.t()
    else:
        raise TypeError(
            f"THIS SHOULD NOT HAPPEN!! Unsupported type {type(adj)} for adjacency matrix."
        )
    return adj
