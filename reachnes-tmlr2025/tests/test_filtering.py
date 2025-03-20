import pytest
import torch
import torch_sparse as tsp
import reachnes.ew_filtering as rn_filter


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("dtype", ["float", "double"])
def test_betainc_filter(sparse_input: bool, dtype: str):
    """Test the default betainc parameters which should be an identity transform."""
    dtype = torch.float if dtype == "float" else torch.double
    bi_filter = rn_filter.BetaincFilter(
        pq_learnable=False, scaling_learnable=False, dtype=dtype
    )
    matrices = torch.rand((2, 10, 10), dtype=dtype)
    if sparse_input:
        mats = tuple(tsp.SparseTensor.from_dense(mat_) for mat_ in matrices)
        filtered = bi_filter(mats)
        filtered = torch.stack([mat.to_dense() for mat in filtered], dim=0)
    else:
        filtered = bi_filter(matrices)

    assert filtered.dtype == dtype
    torch.testing.assert_close(matrices, filtered)
