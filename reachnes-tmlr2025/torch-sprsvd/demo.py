import pytest
import torch
import torch.nn as nn
import torch_sparse as tsp
import tests.utils as tutils
import torch_sprsvd.sprsvd as tsprsvd
import torch_sprsvd.core as tsprsvd_core


def multi_pass_rsvd(
    input_matrix: tsprsvd_core.TORCH_MATRIX,
    k: int,
    num_oversampling: int,
    num_iter: int,
):
    U, singular_values, Vh = tsprsvd.multi_pass_rsvd(
        input_matrix, k, num_oversampling, num_iter=num_iter
    )
    return singular_values


def halko_single_pass_rsvd(
    input_matrix: tsprsvd_core.TORCH_MATRIX, k: int, num_oversampling: int
):
    U, singular_values, Vh = tsprsvd.single_pass_rsvd(input_matrix, k, num_oversampling)
    return singular_values


def block_single_pass_rsvd(
    input_matrix: tsprsvd_core.TORCH_MATRIX,
    k: int,
    num_oversampling: int,
    block_size: int,
):
    U, singular_values, Vh = tsprsvd.single_pass_block_rsvd(
        input_matrix, k, num_oversampling, block_size
    )
    return singular_values


def stream_sp_rsvd(
    input_matrix: tsprsvd_core.TORCH_MATRIX,
    k: int,
    num_oversampling: int,
    block_size: int,
    dtype,
):
    num_cols = input_matrix.size(1)
    sprsvd_pipeline = tsprsvd.StreamSPBlockRSVD.create(
        num_cols=num_cols,
        k=k,
        num_oversampling=num_oversampling,
        block_size=block_size,
        dtype=dtype,
    )

    batch_size = 10
    for inx in range(0, num_cols, batch_size):
        batch = input_matrix[inx : inx + batch_size]
        sprsvd_pipeline.update(batch)
    U, singular_values, Vh = sprsvd_pipeline.compute_block_rsvd()

    return singular_values


def stream_example_with_autograd(
    input_matrix: tsprsvd_core.TORCH_MATRIX,
    k: int,
    num_oversampling: int,
    block_size: int,
    dtype,
):
    num_cols = input_matrix.size(1)
    input_matrix = nn.Parameter(input_matrix)
    sprsvd_pipeline = tsprsvd.StreamSPBlockRSVD.create(
        num_cols=num_cols,
        k=k,
        num_oversampling=num_oversampling,
        block_size=block_size,
        dtype=dtype,
    )

    batch_size = 10
    for inx in range(0, num_cols, batch_size):
        batch = input_matrix[inx : inx + batch_size]
        sprsvd_pipeline.update(batch)
    U, singular_values, Vh = sprsvd_pipeline.compute_block_rsvd()

    loss = torch.sum(singular_values)
    loss.backward()
    print(input_matrix.grad)


def main():
    dtype = "float"
    k = 40
    block_size = 8
    num_oversampling = 8
    type_ = 6
    mat_type = "dense"

    dtype = torch.double if dtype == "double" else torch.float
    num_rows = num_cols = 60
    true_singular_values = tutils.synthetic_singular_values(type_=type_, p=num_cols).to(
        dtype=dtype
    )
    A = tutils.generate_synthetic_matrix(num_cols, num_rows, type_=type_, dtype=dtype)

    input_matrix = tsp.SparseTensor.from_dense(A) if mat_type == "sparse" else A

    sv_multipass = multi_pass_rsvd(
        input_matrix, k=k, num_oversampling=num_oversampling, num_iter=8
    )
    sv_halko = halko_single_pass_rsvd(
        input_matrix, k=k, num_oversampling=num_oversampling
    )
    sv_block_sp = block_single_pass_rsvd(
        input_matrix, k=k, num_oversampling=num_oversampling, block_size=block_size
    )
    sv_stream = stream_sp_rsvd(
        input_matrix,
        k=k,
        num_oversampling=num_oversampling,
        block_size=block_size,
        dtype=dtype,
    )

    print(
        "Multi-pass max abs error: ",
        torch.max(torch.abs(sv_multipass - true_singular_values[:k])).item(),
    )
    print(
        "Halko SP max abs error: ",
        torch.max(torch.abs(sv_halko - true_singular_values[:k])).item(),
    )
    print(
        "Block SP max abs error: ",
        torch.max(torch.abs(sv_block_sp - true_singular_values[:k])).item(),
    )
    print(
        "Stream SP max abs error: ",
        torch.max(torch.abs(sv_stream - true_singular_values[:k])).item(),
    )


if __name__ == "__main__":
    main()
