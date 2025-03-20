import torch
import torch.sparse


def synthetic_singular_values(type_: int, p: int, dtype: torch.dtype = torch.float):
    if type_ == 1:
        sigma1 = torch.pow(torch.tensor([10.0]), -(4 / 19) * torch.arange(0, 20))
        sigma2 = torch.tensor([1e-4]) / (torch.pow(torch.arange(1, p - 20 + 1), 0.1))
        sigma = torch.cat((sigma1, sigma2))
    elif type_ == 2:
        sigma = 1.0 / (torch.arange(1, p + 1) ** 2)
    elif type_ == 3:
        sigma = 1.0 / (torch.arange(1, p + 1) ** 3)
    elif type_ == 4:
        sigma = torch.exp(-torch.arange(1, p + 1) / 7.0).to(dtype=dtype)
    elif type_ == 5:
        sigma = torch.pow(torch.tensor([10.0]), -0.1 * torch.arange(1, p + 1, 1))
    elif type_ == 6:
        sigma1 = torch.cat(
            [
                torch.full(size=(3,), fill_value=1.0, dtype=dtype),
                torch.full(size=(3,), fill_value=0.67, dtype=dtype),
                torch.full(size=(3,), fill_value=0.34, dtype=dtype),
                torch.full(size=(3,), fill_value=0.01, dtype=dtype),
            ]
        )
        sigma2 = torch.tensor([1e-2]) * torch.arange(p - 13, -1, -1) / (p - 13)
        sigma = torch.cat((sigma1, sigma2))
    else:
        raise ValueError(f"Unknown text matrix type '{type_}'")
    sigma = sigma.to(dtype=dtype)
    return sigma


def generate_synthetic_matrix(
    num_rows: int, num_cols: int, type_: int, dtype: torch.dtype = torch.float
):
    p = min(num_rows, num_cols)
    assert p > 20
    L = torch.randn(num_rows, num_rows, dtype=dtype)
    U, _ = torch.linalg.qr(L)
    L = torch.randn(num_cols, num_cols, dtype=dtype)
    V, _ = torch.linalg.qr(L)

    sigma = synthetic_singular_values(type_=type_, p=p, dtype=dtype)
    sigma = torch.sparse.spdiags(
        sigma, offsets=torch.tensor([0]), shape=(num_rows, num_cols)
    )
    matrix = U @ sigma @ V
    return matrix
