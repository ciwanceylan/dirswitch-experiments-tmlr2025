import pytest
import torch
import torch_sparse as tsp
import tests.utils as tutils
import torch_sprsvd.sprsvd as tsprsvd


@pytest.mark.parametrize("type_", range(1, 7))
@pytest.mark.parametrize("k", [1, 5, 10, 20, 50])
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("mat_type", ["dense", "sparse"])
def test_rsvd(type_: int, k: int, dtype: torch.dtype, mat_type: str):
    dtype = torch.double if dtype == "double" else torch.float
    num_rows = num_cols = 50
    true_singular_values = tutils.synthetic_singular_values(type_=type_, p=num_cols).to(
        dtype=dtype
    )
    A = tutils.generate_synthetic_matrix(num_cols, num_rows, type_=type_, dtype=dtype)

    input_matrix = tsp.SparseTensor.from_dense(A) if mat_type == "sparse" else A

    num_iter_sv_errors = []
    num_iter_recon_errors = []
    for num_iter in range(3):
        over_sampling_sv_errors = []
        over_sampling_recon_errors = []
        for i, num_oversampling in enumerate([0, 5, 10, 20]):
            U, sig_values, Vh = tsprsvd.multi_pass_rsvd(
                input_matrix=input_matrix,
                k=k,
                num_oversampling=num_oversampling,
                num_iter=num_iter,
            )

            sv_error = (1.0 / k) * torch.linalg.norm(
                true_singular_values[:k] - sig_values[:k]
            )
            over_sampling_sv_errors.append(sv_error)

            recon = U @ torch.diag(sig_values) @ Vh
            recon_error = torch.linalg.matrix_norm(recon - A, ord="fro")
            over_sampling_recon_errors.append(recon_error)

            # if num_iter > 0:
            #     assert sv_error < num_iter_sv_errors[num_iter - 1][i] + 1e-7
            #     assert recon_error < num_iter_recon_errors[num_iter - 1][i] + 1e-7
            #
            # if i > 0:
            #     assert sv_error < over_sampling_sv_errors[i - 1] + 1e-7
            #     assert recon_error < over_sampling_recon_errors[i - 1] + 1e-7

        num_iter_sv_errors.append(over_sampling_sv_errors)
        num_iter_recon_errors.append(over_sampling_recon_errors)
