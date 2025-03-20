from typing import Optional

import torch
import torch_sparse as tsp
from torch import nn as nn

import reachnes.adj_utils as adjutils
from reachnes import (
    coeffs as rn_coeffs,
    ew_filtering as rn_filtering,
    adj_utils as rn_adjutils,
)
from reachnes.adj_utils import AdjOrientation, AdjSeq
import reachnes.utils as utils


class ReachabilityModel(nn.Module):
    def __init__(
        self,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        ew_filter: Optional[rn_filtering.FilterModel] = None,
    ):
        super().__init__()

        self.coeffs_obj = coeffs_obj
        self.ew_filter = ew_filter

    def forward(
        self,
        adj_obj: rn_adjutils.TorchAdj,
        adj_seq: rn_adjutils.AdjSeq,
        batch_indices: torch.LongTensor,
    ) -> utils.MultiTorchAdjType:
        reachability = compute_reachability(
            adj_obj=adj_obj,
            batch_indices=batch_indices,
            coeffs=self.coeffs_obj(),
            adj_seq=adj_seq,
        )

        if self.ew_filter is not None:
            reachability = self.ew_filter(reachability)

        return reachability


class ReachabilityTimesXModel(nn.Module):
    def __init__(
        self,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        ew_filter: Optional[rn_filtering.FilterModel] = None,
    ):
        super().__init__()

        self.coeffs_obj = coeffs_obj
        self.ew_filter = ew_filter

    def forward(
        self,
        adj_obj: rn_adjutils.TorchAdj,
        adj_seq: rn_adjutils.AdjSeq,
        x: torch.Tensor,
    ) -> torch.Tensor:
        reachability = compute_reachability_times_x(
            adj_obj=adj_obj,
            x=x,
            coeffs=self.coeffs_obj(),
            adj_seq=adj_seq,
        )

        if self.ew_filter is not None:
            reachability = self.ew_filter(reachability)

        return reachability


def adj_matmul(
    adj_obj: adjutils.TorchAdj, matrix: torch.Tensor, orientation: AdjOrientation
) -> torch.Tensor:
    """Helper function for matrix multiplication using the normalized adjacency matrix using different normalizations and orientations."""
    mm_pkg = tsp if adj_obj.is_sparse else torch
    if orientation == "O":
        out = mm_pkg.matmul(adj_obj.AD, matrix)
    elif orientation == "I":
        out = mm_pkg.matmul(adj_obj.DA.t(), matrix)
    elif orientation == "F":
        out = mm_pkg.matmul(adj_obj.DA, matrix)
    elif orientation == "B":
        out = mm_pkg.matmul(adj_obj.AD.t(), matrix)
    elif orientation == "X":
        out = mm_pkg.matmul(adj_obj.DAD, matrix)
    elif orientation == "U":
        out = mm_pkg.matmul(adj_obj.UD, matrix)
    elif orientation == "C":
        out = mm_pkg.matmul(adj_obj.UD.t(), matrix)
    elif orientation == "S":
        out = mm_pkg.matmul(adj_obj.DUD, matrix)
    elif orientation == "A":
        out = mm_pkg.matmul(adj_obj.adj_, matrix)
    elif orientation == "T":
        out = mm_pkg.matmul(adj_obj.adj_.t(), matrix)
    else:
        raise ValueError(f"Unrecognized adjacency matrix orientation '{orientation}'.")
    return out


# @torch.jit.script
def _create_batch(n_nodes: int, node_indices: torch.LongTensor, dtype: torch.dtype):
    if len(node_indices.shape) == 1:
        node_indices = node_indices.unsqueeze(0)
    v = torch.zeros(
        (n_nodes, node_indices.shape[1]), device=node_indices.device, dtype=dtype
    )
    v.scatter_(dim=0, index=node_indices, value=1.0)
    return v


def compute_reachability(
    adj_obj: adjutils.TorchAdj,
    batch_indices: torch.LongTensor,
    coeffs: torch.Tensor,
    adj_seq: AdjSeq,
) -> torch.Tensor:
    """Compute a batch of the polynomial series of the adjacency matrix in `adj_obj` defined by the coefficient in `coeffs`.

    Args:
        adj_obj: An adjacency matrix object which can produce different normalizations.
        batch_indices: Which columns in f(X) to compute.
        coeffs: The Taylor coefficients defining f(X) and the order of the approximation polynomial
        adj_seq:

    Returns:
        out (torch.Tensor): The `batch_indices` columns of f(X)
    """
    device = adj_obj.device
    dtype = adj_obj.dtype

    order = coeffs.shape[1] - 1
    alt_seq = adj_seq.sequence(order=order)
    assert len(alt_seq) == order
    num_tau = coeffs.shape[0]
    batch_indices = batch_indices.to(device)
    monome = _create_batch(adj_obj.num_nodes, batch_indices, dtype=dtype)
    # out = [torch.zeros_like(monome) for _ in range(num_tau)]
    out = torch.zeros(
        (num_tau, monome.shape[0], monome.shape[1]), dtype=dtype, device=device
    )

    out = torch.add(
        out, coeffs[:, 0][:, None, None] * monome[None, :, :]
    )  # Constant in series
    for k, orientation in enumerate(alt_seq):
        monome = adj_matmul(adj_obj=adj_obj, matrix=monome, orientation=orientation)
        out = torch.add(out, coeffs[:, k + 1][:, None, None] * monome[None, :, :])

    return out


def compute_reachability_times_x(
    adj_obj: adjutils.TorchAdj, x: torch.Tensor, coeffs: torch.Tensor, adj_seq: AdjSeq
) -> torch.Tensor:
    """Efficient computation of R @ X.
     But evaluating R @ X for one term at a time in $R$, the complexity reduces from O(Kmn) to O(Kmd).

    Args:
        adj_obj: An adjacency matrix object which can produce different normalizations. [n x n]
        x: Matrix of node features [n x d]
        coeffs: The Taylor coefficients defining the random-walk length probabilities
        adj_seq: Sequence of orientations of the adjacency matrix

    Returns:
        out (torch.Tensor): R @ X [n x d]
    """
    device = x.device
    dtype = x.dtype

    order = coeffs.shape[1] - 1
    alt_seq = adj_seq.sequence(order=order)
    assert len(alt_seq) == order

    monome = x
    out = torch.zeros(
        (coeffs.shape[0], monome.shape[0], monome.shape[1]), dtype=dtype, device=device
    )

    out = torch.add(
        out, coeffs[:, 0][:, None, None] * monome[None, :, :]
    )  # Constant in series
    for k, orientation in enumerate(alt_seq):
        monome = adj_matmul(adj_obj=adj_obj, matrix=monome, orientation=orientation)
        out = torch.add(out, coeffs[:, k + 1][:, None, None] * monome[None, :, :])
    return out
