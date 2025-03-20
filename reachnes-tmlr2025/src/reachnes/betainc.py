from operator import itemgetter
import torch
import torch.special as tspecial


def f_fun(p: torch.Tensor, q: torch.Tensor, x: torch.Tensor):
    return (q / p) * (x / (1 - x))


def a_n_fun(p: torch.Tensor, q: torch.Tensor, f: torch.Tensor, n: int):
    if n == 1:
        a_n = (p * (q - 1) / (q * (p + 1))) * f
    else:
        a_n = f**2 * (
            (p**2 * (n - 1) * (p + q + n - 2) * (p + n - 1) * (q - n))
            / (q**2 * (p + 2 * n - 3) * (p + 2 * n - 2) ** 2 * (p + 2 * n - 1))
        )
    return a_n


def b_n_fun(p: torch.Tensor, q: torch.Tensor, f: torch.Tensor, n: int):
    pf = p * f
    b_n = (2 * (pf + 2 * q) * (n**2 + (p - 1) * n) + p * q * (p - 2 - pf)) / (
        q * (p + 2 * n - 2) * (p + 2 * n)
    )
    return b_n


def k_factor(x: torch.Tensor, p: torch.Tensor, q: torch.Tensor):
    lnbeta = tspecial.gammaln(p) + tspecial.gammaln(q) - tspecial.gammaln(p + q)
    lnK = p * torch.log(x) + (q - 1) * torch.log(1 - x) - torch.log(p) - lnbeta
    return torch.exp(lnK)


def fraction_factor(x: torch.Tensor, p: torch.Tensor, q: torch.Tensor, order: int = 10):
    f = f_fun(p, q, x)
    frac_val = a_n_fun(p, q, f, n=order) / b_n_fun(p, q, f, n=order)
    for n in range(order - 1, 0, -1):
        frac_val += b_n_fun(p, q, f, n=n)
        frac_val = a_n_fun(p, q, f, n=n) / frac_val
    frac_val += 1
    return frac_val


def betainc_frac_sub(x, p, q, order):
    return k_factor(x, p, q) * fraction_factor(x, p, q, order)


def forward_recurrence(
    x: torch.Tensor, p: torch.Tensor, q: torch.Tensor, order: int = 10
):
    f = f_fun(p, q, x)
    A_n_2 = torch.ones(1, dtype=x.dtype, device=x.device)
    A_n_1 = torch.ones(1, dtype=x.dtype, device=x.device)
    B_n_2 = torch.zeros(1, dtype=x.dtype, device=x.device)
    B_n_1 = torch.ones(1, dtype=x.dtype, device=x.device)
    for n in range(1, order + 1):
        a_n = a_n_fun(p, q, f, n)
        b_n = b_n_fun(p, q, f, n)
        A_n = a_n * A_n_2 + b_n * A_n_1
        B_n = a_n * B_n_2 + b_n * B_n_1
        A_n_2 = A_n_1
        B_n_2 = B_n_1
        A_n_1 = A_n
        B_n_1 = B_n
    return A_n_1 / B_n_1


def betainc_fr_sub(x, p, q, order):
    return k_factor(x, p, q) * forward_recurrence(x, p, q, order)


def betainc(x, p, q, order, method="frac"):
    x = torch.as_tensor(x)
    p = torch.as_tensor(p)
    q = torch.as_tensor(q)
    x, p, q = _fix_float_dtype_help(x, p, q)
    x, p, q = _broadcasting_help(x, p, q)

    if method == "frac":

        def betainc_sub(x, p, q, order):
            return k_factor(x, p, q) * fraction_factor(x, p, q, order)
    elif method == "fr":

        def betainc_sub(x, p, q, order):
            return k_factor(x, p, q) * forward_recurrence(x, p, q, order)
    else:
        raise ValueError(f"Unknown betainc method '{method}'.")

    out = torch.zeros_like(x)
    mask = x > p / (p + q)

    out[mask] = 1 - betainc_sub(
        1 - x[mask],
        q[mask] if q.shape == x.shape else q,
        p[mask] if p.shape == x.shape else p,
        order,
    )
    out[~mask] = betainc_sub(
        x[~mask],
        p[~mask] if p.shape == x.shape else p,
        q[~mask] if q.shape == x.shape else q,
        order,
    )
    return out


def _broadcasting_help(*tensors: torch.Tensor):
    largest_shape = max(
        [(tens.shape, torch.numel(tens)) for tens in tensors], key=itemgetter(1)
    )[0]
    tensors = tuple(tens.expand(largest_shape) for tens in tensors)
    return tensors


def _fix_float_dtype_help(*tensors: torch.Tensor):
    dtype = (
        torch.float64
        if any(tens.dtype == torch.float64 for tens in tensors)
        else torch.float32
    )
    tensors = tuple(tens.to(dtype) for tens in tensors)
    return tensors
