from typing import Callable
from functools import partial
import time
import tqdm.auto as tqdm
import pandas as pd
import torch
import tests.utils as test_utils
import torch_sprsvd.sprsvd as sprsvd


def get_svd_methods(k: int):
    svd_methods = {
        "svd": torch.linalg.svd,
        "rsvd": partial(sprsvd.multi_pass_rsvd, k=k, num_oversampling=10, num_iter=1),
        "sp_rsvd_col": partial(
            sprsvd.single_pass_rsvd, k=k, num_oversampling=10, mode="col_projection"
        ),
        "sp_rsvd_combined": partial(
            sprsvd.single_pass_rsvd, k=k, num_oversampling=10, mode="combined"
        ),
        "sp_block_rsvd": partial(
            sprsvd.single_pass_block_rsvd, k=k, num_oversampling=10, block_size=5
        ),
    }
    return svd_methods


def compute_singular_value_error(
    svd_method: Callable, k: int, type_: int, dtype, num_cols: int = 50
):
    num_rows = num_cols
    true_singular_values = test_utils.synthetic_singular_values(
        type_=type_, p=num_cols
    ).to(dtype=dtype)
    A = test_utils.generate_synthetic_matrix(
        num_cols, num_rows, type_=type_, dtype=dtype
    )
    start = time.time()
    U, sig_values, V = svd_method(A)
    duration = time.time() - start
    error = (1.0 / k) * torch.linalg.norm(true_singular_values[:k] - sig_values[:k])
    return error.item(), duration


def compute_errors(dtype: torch.dtype = torch.double):
    k = 20
    num_cols = num_rows = 50
    reps = 5
    results = []
    for type_ in tqdm.trange(1, 7):
        for name, svd_method in tqdm.tqdm(get_svd_methods(k=k).items()):
            for rep in range(reps):
                error, duration = compute_singular_value_error(
                    svd_method=svd_method,
                    k=k,
                    type_=type_,
                    dtype=dtype,
                    num_cols=num_cols,
                )
                results.append(
                    {
                        "method": name,
                        "type": type_,
                        "error": error,
                        "rep": rep,
                        "duration": duration,
                    }
                )
    return pd.DataFrame(results)
