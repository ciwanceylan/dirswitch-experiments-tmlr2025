from typing import List, Optional
import torch
import torch_sprsvd.core as core
from torch_sprsvd.core import TORCH_MATRIX, HALKO_RSVD_MODES


class StreamSPBlockRSVD:
    _g_tensors: List[torch.Tensor]
    G_: Optional[torch.Tensor]
    H_: torch.Tensor

    def __init__(self, omega: torch.Tensor, k: int, block_size: int = 10):
        super().__init__()
        l = omega.shape[1]
        if block_size > l or (l // block_size) != (l / block_size):
            raise ValueError(
                f"'k + num_oversampling' and 'block_size' must be compatible, "
                f"meaning that 'k + num_oversampling' must be a multiple of 'block_size'."
                f"Currently k + num_oversampling = {l} and block_size = {block_size}."
            )
        self.omega = omega
        self.k = k
        self.block_size = block_size
        self._g_tensors = []
        self.G_ = None
        self.H_ = torch.zeros(
            (omega.shape[0], omega.shape[1]), dtype=omega.dtype, device=omega.device
        )
        # self._H_kahn_correction = torch.zeros((omega.shape[0], omega.shape[1]), dtype=omega.dtype, device=omega.device)

    @classmethod
    def create(
        cls,
        num_cols: int,
        k: int,
        num_oversampling: int = 10,
        block_size: int = 10,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        num_oversampling = min(num_cols - k, num_oversampling)
        omega = torch.randn(num_cols, k + num_oversampling, dtype=dtype, device=device)
        return cls(omega=omega, k=k, block_size=block_size)

    def update(self, tensor_batch: TORCH_MATRIX):
        G_, H_ = core.calc_gh_batch(tensor_batch, omega=self.omega)
        self._g_tensors.append(G_)
        self.H_ = self.H_ + H_  # [ num_cols x (k+p) ]

    def merge_g(self):
        self.G_ = torch.cat(self._g_tensors, dim=0)
        return self.G_

    def set_G_and_H(self, G: torch.Tensor, H: torch.Tensor):
        self.G_ = G
        self.H_ = H

    def compute_block_rsvd(self):
        if self.G_ is None:
            self.merge_g()
        return core.gh_sp_rsvd_block(
            omega_cols=self.omega,
            G=self.G_,
            H=self.H_,
            k=self.k,
            block_size=self.block_size,
        )


def multi_pass_rsvd(
    input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, num_iter: int = 0
):
    return core.rsvd_basic(
        input_matrix=input_matrix,
        k=k,
        num_oversampling=num_oversampling,
        num_iter=num_iter,
    )


def single_pass_rsvd(
    input_matrix: TORCH_MATRIX,
    k: int,
    num_oversampling: int = 10,
    mode: HALKO_RSVD_MODES = "col_projection",
):
    return core.sp_rsvd_halko(
        input_matrix=input_matrix, k=k, num_oversampling=num_oversampling, mode=mode
    )


def single_pass_block_rsvd(
    input_matrix: TORCH_MATRIX, k: int, num_oversampling: int = 10, block_size: int = 10
):
    if block_size > (k + num_oversampling):
        raise ValueError(f"Block rsvd requires 'block_size' <= k ")
    return core.sp_rsvd_block(
        input_matrix=input_matrix,
        k=k,
        num_oversampling=num_oversampling,
        block_size=block_size,
    )
