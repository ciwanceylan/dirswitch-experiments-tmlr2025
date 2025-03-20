import pytest
import torch
import numpy as np
import mpmath
import scipy.special as scspecial
import reachnes.betainc as rn_betainc


def p_gradient_analytic(x, p, q):
    mpmath.mp.dps += 5
    der = (
        scspecial.digamma(p + q) - scspecial.digamma(p) + np.log(x)
    ) * scspecial.betainc(p, q, x) - (1 / scspecial.beta(p, q)) * (
        (x**p) / (p**2)
    ) * float(mpmath.hyp3f2(1 - q, p, p, p + 1, p + 1, x))
    mpmath.mp.dps -= 5
    return der


def q_gradient_analytic(x, p, q):
    der = -p_gradient_analytic(1 - x, q, p)
    return der


@pytest.mark.parametrize("pval", [0.01, 1, 100])
@pytest.mark.parametrize("qval", [0.01, 1, 100])
def test_precision(pval, qval):
    """Test how close pytorch implementations are to scipy implementation."""
    dtype = torch.float32
    order = 15
    x = torch.arange(0.01, 0.99, 0.01, dtype=dtype)
    p = torch.tensor([pval], dtype=dtype, requires_grad=False)
    q = torch.tensor([qval], dtype=dtype, requires_grad=False)
    y_frac = rn_betainc.betainc(x, p, q, order, method="frac").numpy()
    y_forward_rec = rn_betainc.betainc(x, p, q, order, method="fr").numpy()
    y_true = scspecial.betainc(p.item(), q.item(), x.numpy())
    assert y_frac.dtype == y_true.dtype
    np.testing.assert_allclose(y_frac, y_true, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(y_forward_rec, y_true, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("xval", [0.02, 0.4, 0.6, 0.99])
@pytest.mark.parametrize("pval", [0.01, 1, 100])
@pytest.mark.parametrize("qval", [0.01, 1, 100])
# @pytest.mark.parametrize('method', ['frac', 'fr'])
def test_precision_gradient(xval, pval, qval):
    """Test how close autograd is to numerical gradient.
    Note that numerical gradient is not so accurate itself, so this only tests if they are roughly similar.
    We are only testing method 'frac' since they perform identically on the precision test."""
    dtype = torch.float32
    order = 15
    step_size = 0.0007
    steps = 10
    x = torch.tensor([xval], dtype=dtype, requires_grad=False)

    """ Test p gradient """
    pvals = torch.arange(pval - steps * step_size, pval + steps * step_size, step_size)
    q = torch.tensor([qval], dtype=dtype, requires_grad=False)
    y_frac = []
    grads = []
    for pval_ in pvals:
        p = torch.tensor([pval_], dtype=dtype, requires_grad=True)
        y_frac_val = rn_betainc.betainc(x, p, q, order, method="frac")
        y_frac_val.backward()
        grads.append(p.grad.item())
        y_frac.append(y_frac_val.item())

    nummerical_grad = torch.gradient(torch.tensor(y_frac), spacing=step_size)[0].numpy()
    # The ends are excluded from the test because the numerical gradient is less accurate there
    if pval > 10 or qval > 10 or (pval > 10 * qval):
        # When one value is large or p is much larger the q, the numerical gradient is particularly bad, so we increase tolerance.
        np.testing.assert_allclose(
            grads[3:-3], nummerical_grad[3:-3], atol=1e-2, rtol=5e-2
        )
    else:
        np.testing.assert_allclose(
            grads[3:-3], nummerical_grad[3:-3], atol=1e-4, rtol=1e-2
        )

    """ Test q gradient """
    qvals = torch.arange(qval - steps * step_size, qval + steps * step_size, step_size)
    p = torch.tensor([pval], dtype=dtype, requires_grad=False)
    y_frac = []
    grads = []
    for qval_ in qvals:
        q = torch.tensor([qval_], dtype=dtype, requires_grad=True)
        y_frac_val = rn_betainc.betainc(x, p, q, order, method="frac")
        y_frac_val.backward()
        grads.append(q.grad.item())
        y_frac.append(y_frac_val.item())

    nummerical_grad = torch.gradient(torch.tensor(y_frac), spacing=step_size)[0].numpy()
    # The ends are excluded from the test because the numerical gradient is less accurate there
    if pval > 10 or qval > 10 or (pval > 10 * qval):
        # When one value is large or p is much larger the q, the numerical gradient is particularly bad, so we increase tolerance.
        np.testing.assert_allclose(
            grads[3:-3], nummerical_grad[3:-3], atol=1e-2, rtol=5e-2
        )
    else:
        np.testing.assert_allclose(
            grads[3:-3], nummerical_grad[3:-3], atol=1e-4, rtol=1e-2
        )


@pytest.mark.parametrize("xval", [0.02, 0.4, 0.5, 0.6, 0.99])
@pytest.mark.parametrize("pval", [0.01, 1, 100])
@pytest.mark.parametrize("qval", [0.01, 1, 100])
# @pytest.mark.parametrize('method', ['frac', 'fr'])
def test_precision_gradient(xval, pval, qval):
    """Test how autograd against analytical gradients."""
    dtype = torch.float32
    order = 20

    x = torch.tensor([xval], dtype=dtype, requires_grad=False)
    p = torch.tensor([pval], requires_grad=True, dtype=dtype)
    q = torch.tensor([qval], requires_grad=True, dtype=dtype)
    y_frac_val = rn_betainc.betainc(x, p, q, order, method="fr")
    y_frac_val.sum().backward()
    p_auto_der = p.grad.item()
    q_auto_der = q.grad.item()

    p_anal_der = p_gradient_analytic(xval, pval, qval)
    q_anal_der = q_gradient_analytic(xval, pval, qval)
    np.testing.assert_allclose(
        p_auto_der, np.asarray(p_anal_der, dtype=np.float32), atol=1e-5, rtol=1e-3
    )
    np.testing.assert_allclose(
        q_auto_der, np.asarray(q_anal_der, dtype=np.float32), atol=1e-5, rtol=1e-3
    )
