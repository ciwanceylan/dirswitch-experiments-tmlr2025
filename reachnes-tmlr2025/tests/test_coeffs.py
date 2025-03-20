from typing import Dict
import pytest
import numpy as np
import scipy.stats as sstats
import torch

import reachnes.coeffs as rn_coeffs

COEFF_SPECS = [
    rn_coeffs.CoeffsSpec(name="poisson", kwargs={"tau": 2.0}, loc=0),
    rn_coeffs.CoeffsSpec(name="poisson", kwargs={"tau": 2.0}, loc=2),
    rn_coeffs.CoeffsSpec(name="poisson", kwargs={"tau": 3.0}, loc=2),
    rn_coeffs.CoeffsSpec(name="geometric", kwargs={"tau": 2.0}, loc=0),
    rn_coeffs.CoeffsSpec(name="geometric", kwargs={"tau": 5.0}, loc=1),
    rn_coeffs.CoeffsSpec(name="geometric", kwargs={"alpha": 0.8}, loc=0),
    rn_coeffs.CoeffsSpec(name="geometric", kwargs={"alpha": 0.1}, loc=2),
    rn_coeffs.CoeffsSpec(name="binom", kwargs={"tau": 2.0}, loc=0),
    rn_coeffs.CoeffsSpec(name="binom", kwargs={"tau": 5.0}, loc=0),
    rn_coeffs.CoeffsSpec(name="binom", kwargs={"alpha": 0.8}, loc=0),
    rn_coeffs.CoeffsSpec(name="binom", kwargs={"alpha": 0.99}, loc=1),
    rn_coeffs.CoeffsSpec(name="binom", kwargs={"alpha": 0.1}, loc=2),
    rn_coeffs.CoeffsSpec(name="nbinom", kwargs={"tau": 2.0, "r": 2}, loc=0),
    rn_coeffs.CoeffsSpec(name="nbinom", kwargs={"alpha": 0.3, "r": 5}, loc=0),
    rn_coeffs.CoeffsSpec(name="nbinom", kwargs={"alpha": 0.99999, "r": 3}, loc=0),
    rn_coeffs.CoeffsSpec(name="nbinom", kwargs={"alpha": 0.00001, "r": 3}, loc=0),
    rn_coeffs.CoeffsSpec(name="nbinom", kwargs={"alpha": 0.001, "r": 3}, loc=2),
    rn_coeffs.CoeffsSpec(name="uniform", kwargs={"tau": 1}, loc=0),
    rn_coeffs.CoeffsSpec(name="uniform", kwargs={"tau": 1}, loc=1),
    rn_coeffs.CoeffsSpec(name="uniform", kwargs={"tau": 3}, loc=2),
]


def scipy_distributions(name: str, order: int, loc: int, kwargs: Dict[str, float]):
    k = np.arange(0, order + 1)
    if name == "poisson":
        pmf = sstats.poisson.pmf(k, mu=kwargs["tau"], loc=loc)
    elif name == "geometric":
        alpha = (
            kwargs["alpha"]
            if "alpha" in kwargs
            else kwargs["tau"] / (1.0 + kwargs["tau"])
        )
        pmf = sstats.geom.pmf(k, p=1.0 - alpha, loc=loc - 1)
    elif name == "uniform":
        pmf = sstats.randint.pmf(k=k, low=0, high=2 * kwargs["tau"] + 1, loc=loc)
        pmf = pmf / pmf.sum()
    elif name == "nbinom":
        alpha = (
            kwargs["alpha"]
            if "alpha" in kwargs
            else kwargs["tau"] / (kwargs["r"] + kwargs["tau"])
        )
        pmf = sstats.nbinom.pmf(k=k, n=kwargs["r"], p=1.0 - alpha, loc=loc)
    elif name == "binom":
        n = order + 1 - loc
        p = 1 - kwargs["alpha"] if "alpha" in kwargs else kwargs["tau"] / n
        p = min(p, 1 - 1e-6)  # Protect against the case where tau > n
        pmf = sstats.binom.pmf(k=k, n=n, p=p, loc=loc)
    else:
        raise NotImplementedError(
            f"Unknown distribution '{name}' with kwargs: {kwargs}"
        )
    return pmf


@pytest.mark.parametrize("coeff_spec", COEFF_SPECS)
@pytest.mark.parametrize("order", [3, 5, 10])
def test_coefficients(coeff_spec, order):
    coeffs = rn_coeffs.coeffs_from_name(
        name=coeff_spec.name,
        rwl_distri_kwargs=coeff_spec.kwargs,
        order=order,
        loc=coeff_spec.loc,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    gt_coeffs = scipy_distributions(
        name=coeff_spec.name, order=order, loc=coeff_spec.loc, kwargs=coeff_spec.kwargs
    )
    np.testing.assert_allclose(coeffs.numpy(), gt_coeffs)


@pytest.mark.parametrize("order", [2, 5, 10])
def test_coefficients_model(order):
    coeffs = rn_coeffs.RWLCoefficientsModel.from_rwl_distributions(
        COEFF_SPECS,
        order=order,
        normalize=True,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert coeffs.num_series == len(COEFF_SPECS)
    assert coeffs.num_coeffs == order + 1
    coefficients = coeffs()
    torch.testing.assert_allclose(
        torch.sum(coefficients, dim=1), torch.ones(coeffs.num_series)
    )
