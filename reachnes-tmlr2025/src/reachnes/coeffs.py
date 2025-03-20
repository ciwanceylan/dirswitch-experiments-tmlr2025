from typing import Dict, Sequence, Union, Literal, List
import dataclasses as dc
import numpy as np
import scipy.special as sspec
import scipy.stats as sstats
import torch
from torch import nn as nn

RWL_DISTR = Literal["geometric", "poisson", "uniform", "nbinom", "binom"]


@dc.dataclass(frozen=True)
class CoeffsSpec:
    name: RWL_DISTR
    kwargs: Dict[str, Union[float, Sequence[float]]]
    loc: int = 0


class RWLCoefficientsModel(nn.Module):
    coeffs_: torch.Tensor
    normalize: bool

    def __init__(self, coeffs: torch.Tensor, normalize: bool):
        super().__init__()
        assert len(coeffs.shape) == 2
        self.register_buffer("coeffs_", coeffs)
        # self.coeffs = coeffs
        self.normalize = normalize

    @classmethod
    def from_rwl_distributions(
        cls,
        rwl_distrs: Sequence[CoeffsSpec],
        order: int,
        normalize: bool,
        dtype: torch.dtype,
        device: torch.device,
    ):
        num_coeffs = order + 1
        coeffs = coeffs_from_rwl_names(
            rwl_distrs=rwl_distrs, order=order, dtype=dtype, device=device
        )
        assert coeffs.shape[1] == num_coeffs
        assert coeffs.shape[0] == len(rwl_distrs)
        return cls(coeffs=coeffs, normalize=normalize)

    @property
    def num_series(self):
        return self.coeffs_.shape[0]

    @property
    def num_coeffs(self):
        return self.coeffs_.shape[1]

    def forward(self):
        coeffs = self.coeffs_
        if self.normalize:
            coeffs = coeffs / torch.linalg.vector_norm(
                coeffs, ord=1, keepdims=True, dim=1
            )
        return coeffs


def coeffs_from_rwl_names(
    rwl_distrs: Sequence[CoeffsSpec],
    order: int,
    dtype: torch.dtype,
    device: torch.device,
):
    coeffs = []
    for spec in rwl_distrs:
        coeffs.append(
            coeffs_from_name(
                spec.name,
                rwl_distri_kwargs=spec.kwargs,
                loc=spec.loc,
                order=order,
                dtype=dtype,
                device=device,
            )
        )
    coeffs = torch.stack(coeffs, dim=0)
    return coeffs


def coeffs_from_name(
    name: RWL_DISTR,
    rwl_distri_kwargs: Dict[str, Union[float, int]],
    order: int,
    loc: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    num_coeffs = order + 1 - loc
    if num_coeffs < 0:
        raise ValueError(
            f"Requirement 'order + 1 >= loc' not met: order={order}, loc={loc}"
        )

    if name == "geometric" and "alpha" in rwl_distri_kwargs:
        coeffs = geometric_coefficients(
            num_coeffs=num_coeffs,
            alpha=rwl_distri_kwargs["alpha"],
            dtype=dtype,
            device=device,
        )
    elif name == "geometric" and "tau" in rwl_distri_kwargs:
        coeffs = geometric_coefficients_mean(
            num_coeffs=num_coeffs,
            tau=rwl_distri_kwargs["tau"],
            dtype=dtype,
            device=device,
        )
    elif name == "poisson":
        coeffs = poisson_coefficients(
            tau=rwl_distri_kwargs["tau"],
            num_coeffs=num_coeffs,
            dtype=dtype,
            device=device,
        )
    elif name == "uniform":
        coeffs = uniform_coefficients(
            num_coeffs=num_coeffs,
            tau=rwl_distri_kwargs["tau"],
            dtype=dtype,
            device=device,
        )
    elif name == "nbinom" and "alpha" in rwl_distri_kwargs:
        coeffs = neg_binomial_coefficients_alpha(
            num_coeffs=num_coeffs,
            r=rwl_distri_kwargs["r"],
            alpha=rwl_distri_kwargs["alpha"],
            dtype=dtype,
            device=device,
        )
    elif name == "nbinom" and "tau" in rwl_distri_kwargs:
        coeffs = neg_binomial_coefficients_tau(
            num_coeffs=num_coeffs,
            r=rwl_distri_kwargs["r"],
            tau=rwl_distri_kwargs["tau"],
            dtype=dtype,
            device=device,
        )
    elif name == "binom" and "alpha" in rwl_distri_kwargs:
        coeffs = binomial_coefficients(
            alpha=rwl_distri_kwargs["alpha"],
            num_coeffs=num_coeffs,
            dtype=dtype,
            device=device,
        )
    elif name == "binom" and "tau" in rwl_distri_kwargs:
        coeffs = binomial_coefficients_mean(
            tau=rwl_distri_kwargs["tau"],
            num_coeffs=num_coeffs,
            dtype=dtype,
            device=device,
        )
    else:
        raise NotImplementedError(f"Unknown coefficients input '{name}'.")

    if loc > 0:
        coeffs = torch.cat((torch.zeros((loc,), dtype=dtype, device=device), coeffs))
    return coeffs


def uniform_coefficients(
    tau: int, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    nnz = min(int(2 * tau + 1), num_coeffs)
    nnz_coeffs = torch.ones(size=(nnz,), dtype=dtype, device=device)
    num_remaining = num_coeffs - nnz
    remaining = torch.zeros(size=(num_remaining,), dtype=dtype, device=device)
    coeffs = torch.cat((nnz_coeffs, remaining))
    coeffs = coeffs / coeffs.sum()
    return coeffs


def geometric_coefficients(
    alpha: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    if alpha < 1e-7:
        coeffs = torch.zeros(num_coeffs, dtype=dtype, device=device)
        coeffs[0] = 1.0
    else:
        alpha = torch.tensor([alpha], dtype=dtype, device=device)
        coeffs = (1 - alpha) * torch.exp(
            torch.arange(0, num_coeffs, dtype=dtype, device=device) * torch.log(alpha)
        )
    return coeffs


def geometric_coefficients_mean(
    tau: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    alpha = tau / (1.0 + tau)
    return geometric_coefficients(
        alpha, num_coeffs=num_coeffs, dtype=dtype, device=device
    )


def poisson_coefficients(
    tau: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    if tau < 1e-7:
        pois_coeffs = torch.zeros(num_coeffs, dtype=dtype, device=device)
        pois_coeffs[0] = 1.0
    else:
        tau = torch.tensor([tau], dtype=dtype, device=device)
        k_vals = torch.arange(0, num_coeffs, dtype=dtype, device=device)
        log_k_vals = torch.log(
            torch.maximum(k_vals, torch.ones(1, dtype=dtype, device=device))
        )
        log_k_factorial = torch.cumsum(log_k_vals, dim=0)
        log_coeffs = -tau + k_vals * torch.log(tau) - log_k_factorial
        pois_coeffs = torch.exp(log_coeffs)
    return pois_coeffs


def neg_binomial_coefficients_tau(
    r: float, tau: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    alpha = tau / (r + tau)
    return neg_binomial_coefficients_alpha(
        r=r, alpha=alpha, num_coeffs=num_coeffs, dtype=dtype, device=device
    )


def neg_binomial_coefficients_alpha(
    r: float, alpha: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    k_vals = np.arange(0, num_coeffs, dtype=np.float64)
    if alpha < 1e-8:
        out = torch.zeros((num_coeffs,), dtype=dtype)
        out[0] = 1.0
    # elif alpha >= 1 - (1. / num_coeffs):
    #     out = torch.full((num_coeffs,), fill_value=1. / num_coeffs, dtype=dtype)
    else:
        log_coeff = (
            r * np.log((1 - alpha))
            + k_vals * np.log(alpha)
            + sspec.loggamma(k_vals + r)
            - sspec.loggamma(r)
            - sspec.loggamma(k_vals + 1)
        )
        out = torch.from_numpy(np.exp(log_coeff)).to(dtype=dtype, device=device)
    return out


def binomial_coefficients(
    alpha: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    k_vals = np.arange(0, num_coeffs, dtype=np.int64)
    out = sstats.binom.pmf(k=k_vals, n=num_coeffs, p=1.0 - alpha)
    out = torch.from_numpy(out).to(dtype=dtype, device=device)
    return out


def binomial_coefficients_mean(
    tau: float, num_coeffs: int, dtype: torch.dtype, device: torch.device
):
    p = min(tau / num_coeffs, 1.0 - 1e-6)
    return binomial_coefficients(
        alpha=1.0 - p, num_coeffs=num_coeffs, dtype=dtype, device=device
    )
