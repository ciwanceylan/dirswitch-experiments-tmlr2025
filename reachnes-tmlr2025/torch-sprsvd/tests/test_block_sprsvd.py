import itertools
import pytest
import torch
import torch_sparse as tsp
import tests.utils as tutils
import torch_sprsvd.sprsvd as tsprsvd
import torch_sprsvd.core as tsprsvd_core


@pytest.mark.parametrize("type_", range(1, 7))
@pytest.mark.parametrize("k", [1, 5, 10, 20, 50])
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("mat_type", ["dense", "sparse"])
def test_block_rsvd(type_: int, k: int, dtype: torch.dtype, mat_type: str):
    dtype = torch.double if dtype == "double" else torch.float
    num_rows = num_cols = 50
    true_singular_values = tutils.synthetic_singular_values(type_=type_, p=num_cols).to(
        dtype=dtype
    )
    A = tutils.generate_synthetic_matrix(num_cols, num_rows, type_=type_, dtype=dtype)

    input_matrix = tsp.SparseTensor.from_dense(A) if mat_type == "sparse" else A

    for block_size in [1, 2, 5, 10, 20]:
        over_sampling_sv_errors = []
        over_sampling_recon_errors = []
        for i, num_oversampling in enumerate([0, 5, 10, 20]):
            if (k + num_oversampling) < block_size:
                with pytest.raises(ValueError):
                    _ = tsprsvd.single_pass_block_rsvd(
                        input_matrix=input_matrix,
                        k=k,
                        num_oversampling=num_oversampling,
                        block_size=block_size,
                    )

            # elif ((k + num_oversampling) // block_size) != ((k + num_oversampling) / block_size):
            #     with pytest.raises(AssertionError):
            #         _ = tsprsvd.single_pass_block_rsvd(input_matrix=input_matrix, k=k,
            #                                            num_oversampling=num_oversampling,
            #                                            block_size=block_size)
            else:
                U, sig_values, Vh = tsprsvd.single_pass_block_rsvd(
                    input_matrix=input_matrix,
                    k=k,
                    num_oversampling=num_oversampling,
                    block_size=block_size,
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


@pytest.mark.parametrize("type_", range(1, 7))
@pytest.mark.parametrize("k", [10, 50])
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("mat_type", ["dense", "sparse"])
def test_stream_block_rsvd(type_: int, k: int, dtype: torch.dtype, mat_type: str):
    dtype = torch.double if dtype == "double" else torch.float
    num_rows = num_cols = 50
    batch_size = 11
    A = tutils.generate_synthetic_matrix(num_cols, num_rows, type_=type_, dtype=dtype)

    input_matrix = tsp.SparseTensor.from_dense(A) if mat_type == "sparse" else A

    num_oversampling = 10
    omega = torch.randn(
        num_cols, k + num_oversampling, dtype=dtype, device=torch.device("cpu")
    )
    G = input_matrix @ omega
    H = input_matrix.t() @ G

    for block_size in [1, 2, 10]:
        U, sig_values, Vh = tsprsvd_core.gh_sp_rsvd_block(
            omega_cols=omega, G=G, H=H, k=k, block_size=block_size
        )
        rsvd_obj_fixed = tsprsvd.StreamSPBlockRSVD(
            omega=omega, k=k, block_size=block_size
        )
        rsvd_obj_fixed.set_G_and_H(G=G, H=H)
        U_stream, sig_values_stream, Vh_stream = rsvd_obj_fixed.compute_block_rsvd()
        assert torch.allclose(U_stream, U)
        assert torch.allclose(sig_values, sig_values_stream)
        assert torch.allclose(Vh, Vh_stream)

        rsvd_obj = tsprsvd.StreamSPBlockRSVD(omega=omega, k=k, block_size=block_size)
        for index in range(0, num_rows, batch_size):
            start = index
            stop = min(start + batch_size, num_rows)
            batch_indices = torch.arange(start, stop, dtype=torch.long)
            rsvd_obj.update(input_matrix[batch_indices, :])

        rsvd_obj.merge_g()
        torch.testing.assert_close(rsvd_obj.H_, H, rtol=1e-05, atol=1e-07)
        torch.testing.assert_close(rsvd_obj.G_, G)

        _ = rsvd_obj.compute_block_rsvd()
