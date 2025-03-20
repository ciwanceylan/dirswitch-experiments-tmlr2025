from typing import Optional
import dataclasses as dc
import numpy as np
import torch
import torch.nn as nn
import torch_sparse as tsp

import reachnes.betainc as rn_betainc
import reachnes.utils as rn_utils


class FilterModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_: rn_utils.MultiTorchAdjType) -> rn_utils.MultiTorchAdjType:
        return input_


class ThresholdFilter(FilterModel):
    def __init__(self, threshold: float, dense2sparse: bool):
        super().__init__()
        self.threshold = threshold
        self.dense2sparse = dense2sparse

    @classmethod
    def create_from_graph_size(cls, num_nodes: int, num_edges: int, dense2sparse: bool):
        threshold = 1.0 / np.sqrt((num_nodes + 1) * (num_edges + 1))
        return cls(threshold=threshold, dense2sparse=dense2sparse)

    def forward(self, input_: rn_utils.MultiTorchAdjType) -> rn_utils.MultiTorchAdjType:
        if isinstance(input_, torch.Tensor):
            filtered_input = threshold_reachability_dense(
                input_, threshold=self.threshold, dense2sparse=self.dense2sparse
            )
        else:
            assert isinstance(input_, tuple) and isinstance(input_[0], tsp.SparseTensor)
            output = []
            for mat in input_:
                filtered_mat = threshold_reachability_sparse(
                    mat, threshold=self.threshold
                )
                output.append(filtered_mat)
            filtered_input = tuple(output)
        return filtered_input


def threshold_reachability_sparse(
    reachability: tsp.SparseTensor, threshold: float
) -> tsp.SparseTensor:
    theta = torch.tensor(
        threshold, device=reachability.device(), dtype=reachability.dtype()
    )
    values = reachability.storage.value()
    values[torch.lt(torch.abs(values), theta)] = 0.0
    return reachability


def threshold_reachability_dense(
    reachability: torch.Tensor, threshold: float, dense2sparse: bool
) -> rn_utils.MultiTorchAdjType:
    theta = torch.atleast_2d(
        torch.tensor(threshold, device=reachability.device, dtype=reachability.dtype)
    )
    reachability[torch.lt(torch.abs(reachability), theta)] = 0.0
    if dense2sparse:
        reachability = tuple(
            tsp.SparseTensor.from_dense(series) for series in reachability
        )
    return reachability


class LogFilter(FilterModel):
    def __init__(
        self,
        *,
        scaling_factor: float,
        threshold_filter: Optional[ThresholdFilter] = None,
    ):
        super().__init__()
        self.scaling_factor = float(scaling_factor)
        self.threshold_filter = threshold_filter

    def forward(self, input_: rn_utils.MultiTorchAdjType):
        if isinstance(input_, torch.Tensor):
            output = self._log_filter(input_, self.scaling_factor)
        else:
            assert isinstance(input_, tuple) and isinstance(input_[0], tsp.SparseTensor)
            output = []
            for mat in input_:
                filtered_mat = self._log_filter_sparse(mat, self.scaling_factor)
                output.append(filtered_mat)
            output = tuple(output)

        if self.threshold_filter:
            output = self.threshold_filter(output)

        return output

    @staticmethod
    def _log_filter_sparse(input_: tsp.SparseTensor, scaling_factor: float):
        return input_.set_value(
            LogFilter._log_filter(input_.storage.value(), scaling_factor=scaling_factor)
        )

    @staticmethod
    def _log_filter(input_: torch.Tensor, scaling_factor: float):
        scaling_factor = torch.tensor(
            scaling_factor, dtype=input_.dtype, device=input_.device
        )
        log_scaling_factor = torch.log(scaling_factor)
        x = torch.maximum(input_, 1.0 / scaling_factor)
        output = torch.log(x) + log_scaling_factor
        return output


class BetaincFilter(FilterModel):
    def __init__(
        self,
        *,
        dtype: torch.dtype,
        pq_learnable: bool = False,
        scaling_learnable: bool = False,
        init_log_p: float = 0.0,
        init_log_q: float = 0.0,
        init_log_scaling: float = 0.0,
        threshold_filter: Optional[ThresholdFilter] = None,
    ):
        super().__init__()
        init_log_scaling = torch.tensor([init_log_scaling], dtype=dtype)
        init_log_p = torch.tensor([init_log_p], dtype=dtype)
        init_log_q = torch.tensor([init_log_q], dtype=dtype)
        self.threshold_filter = threshold_filter

        if pq_learnable:
            self.log_p = nn.Parameter(init_log_p) if pq_learnable else init_log_p
            self.log_q = nn.Parameter(init_log_q)
        else:
            self.register_buffer("log_p", init_log_p)
            self.register_buffer("log_q", init_log_q)

        if scaling_learnable:
            self.log_scaling = nn.Parameter(init_log_scaling)
        else:
            self.register_buffer("log_scaling", init_log_scaling)

    def forward(self, input_: rn_utils.MultiTorchAdjType):
        p = torch.exp(self.log_p)
        q = torch.exp(self.log_q)
        s = torch.exp(self.log_scaling)
        if isinstance(input_, torch.Tensor):
            output = self._betainc_filter_dense(input_, p=p, q=q, s=s)
        else:
            assert isinstance(input_, tuple) and isinstance(input_[0], tsp.SparseTensor)
            output = []
            for mat in input_:
                filtered_mat = self._betainc_filter_sparse(mat, p=p, q=q, s=s)
                output.append(filtered_mat)
            output = tuple(output)

        if self.threshold_filter:
            output = self.threshold_filter(output)
        return output

    @staticmethod
    def _betainc_filter_sparse(
        input_: tsp.SparseTensor, p: torch.Tensor, q: torch.Tensor, s: torch.Tensor
    ):
        return input_.set_value(
            s * rn_betainc.betainc(input_.storage.value(), p=p, q=q, order=15)
        )

    @staticmethod
    def _betainc_filter_dense(
        input_: torch.Tensor, p: torch.Tensor, q: torch.Tensor, s: torch.Tensor
    ):
        return s * rn_betainc.betainc(input_, p=p, q=q, order=15)
