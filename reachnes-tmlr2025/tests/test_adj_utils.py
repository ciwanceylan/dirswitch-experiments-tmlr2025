import pytest
import numpy as np
import networkx as nx
import torch
import torch_sparse as tsp

import reachnes.adj_utils as adj_utils
from tests.utils import get_tsp_element, get_nx_degrees


class TestAdjSeq:
    def test_get_sequence(self):
        adj_seq = adj_utils.AdjSeq("OSI")
        assert adj_seq.sequence(6) == ("O", "S", "I", "I", "I", "I")

    def test_post_init_empty(self):
        with pytest.raises(AssertionError):
            _ = adj_utils.AdjSeq("")

    def test_post_init_bad_sequence(self):
        with pytest.raises(AssertionError):
            _ = adj_utils.AdjSeq("SY")


@pytest.mark.parametrize("input_class", ["scipy", "torch", "tsp", "numpy"])
@pytest.mark.parametrize("dtype", ["float", "double"])
def test_conversion_to_torch(
    dir_ms_adj_mat: tsp.SparseTensor, input_class: str, dtype: str
):
    dtype = torch.float if dtype == "float" else torch.double
    if input_class == "scipy":
        mat = dir_ms_adj_mat.to_scipy()
    elif input_class == "numpy":
        mat = dir_ms_adj_mat.to_dense().numpy()
    elif input_class == "torch":
        mat = dir_ms_adj_mat.to_dense()
    elif input_class == "tsp":
        mat = dir_ms_adj_mat
    else:
        raise ValueError("Unimplemented test")

    torch_adj = adj_utils.to_torch_adj(mat, dtype=dtype, remove_self_loops=False)
    if input_class in ["torch", "numpy"]:
        assert torch_adj.dtype == dtype
    else:
        assert torch_adj.dtype() == dtype

    dir_ms_adj_mat = dir_ms_adj_mat.to(dtype=dtype)

    if input_class in ["torch", "numpy"]:
        assert torch.allclose(dir_ms_adj_mat.to_dense(), torch_adj)
    else:
        assert torch.allclose(dir_ms_adj_mat.to_dense(), torch_adj.to_dense())


def test_make_undirected(dir_ms_adj_mat: tsp.SparseTensor):
    assert not torch.allclose(dir_ms_adj_mat.to_dense(), dir_ms_adj_mat.to_dense().t())
    dense_undir = adj_utils.to_symmetric(dir_ms_adj_mat.to_dense())
    sparse_undir = adj_utils.to_symmetric(dir_ms_adj_mat)
    assert torch.allclose(dense_undir, dense_undir.t())
    assert torch.allclose(sparse_undir.to_dense(), sparse_undir.to_dense().t())


def test_remove_self_loops(dir_ms_adj_mat: tsp.SparseTensor):
    assert get_tsp_element(dir_ms_adj_mat, 301, 301) > 0
    adj = adj_utils.remove_diag(dir_ms_adj_mat)
    assert get_tsp_element(adj, 301, 301) == 0


def test_remove_self_loops_dense(dir_ms_adj_mat: tsp.SparseTensor):
    dir_adj_mat = dir_ms_adj_mat.to_dense()
    assert dir_adj_mat[301, 301] > 0
    adj = adj_utils.remove_diag(dir_adj_mat)
    assert adj[301, 301] == 0


@pytest.mark.parametrize(
    "in_degrees,weighted", [(False, True), (False, False), (True, True), (True, False)]
)
def test_degrees(
    dir_ms_graph: nx.DiGraph,
    dir_ms_adj_mat: tsp.SparseTensor,
    in_degrees: bool,
    weighted: bool,
):
    degs = adj_utils.calc_degrees(
        dir_ms_adj_mat, in_degrees=in_degrees, weighted=weighted
    )
    for node, deg in get_nx_degrees(
        dir_ms_graph, in_degrees=in_degrees, weighted=weighted
    ):
        assert degs[node] == pytest.approx(deg)

    degs_dense = adj_utils.calc_degrees(
        dir_ms_adj_mat.to_dense(), in_degrees=in_degrees, weighted=weighted
    )
    assert np.allclose(degs, degs_dense)
    degs_tsp = adj_utils.calc_degrees(
        dir_ms_adj_mat, in_degrees=in_degrees, weighted=weighted
    )
    assert np.allclose(degs, degs_tsp)


class TestSymNormalizedAdj:
    def test_normalized_sparse_undir(self, undir_ms_adj_mat: tsp.SparseTensor):
        """Test the symmetric normalization by checking each element."""
        out_degrees = adj_utils.calc_out_degrees(undir_ms_adj_mat, weighted=True)
        in_degrees = adj_utils.calc_in_degrees(undir_ms_adj_mat, weighted=True)
        norm_adj = adj_utils.sym_adj_normalization_tsp(undir_ms_adj_mat)

        num_nnz_diag = ((out_degrees + in_degrees) == 0).sum()
        assert norm_adj.nnz() == undir_ms_adj_mat.nnz() + num_nnz_diag

        for i, j, w in zip(
            norm_adj.storage.row(), norm_adj.storage.col(), norm_adj.storage.value()
        ):
            i = i.item()
            j = j.item()
            denominator = np.sqrt(in_degrees[i] * out_degrees[j])
            value = (
                0
                if not denominator > 0
                else undir_ms_adj_mat[i, j].to_dense() / denominator
            )
            if i == j and denominator == 0:
                assert norm_adj[i, j].to_dense() == pytest.approx(1)
            else:
                assert norm_adj[i, j].to_dense() == pytest.approx(value)

    def test_normalized_dense_undir(self, undir_ms_adj_mat: tsp.SparseTensor):
        """This test assumes that the sparse test above passes."""
        norm_sparse_adj = adj_utils.sym_adj_normalization_tsp(undir_ms_adj_mat)
        norm_dense_adj = adj_utils.sym_adj_normalization_dense(
            undir_ms_adj_mat.to_dense()
        )
        assert torch.allclose(norm_dense_adj, norm_sparse_adj.to_dense())


class TestRWNormalizedAdj:
    @pytest.mark.parametrize("use_in_degrees", [False, True])
    def test_normalized_out(
        self, dir_ms_adj_mat: tsp.SparseTensor, use_in_degrees: bool
    ):
        degrees = adj_utils.calc_degrees(
            dir_ms_adj_mat, weighted=True, in_degrees=use_in_degrees
        )
        norm_adj = adj_utils.rw_adj_normalization_tsp(
            dir_ms_adj_mat, use_out_degrees=not use_in_degrees
        )

        num_nnz_diag = (degrees == 0).sum()
        assert norm_adj.nnz() == dir_ms_adj_mat.nnz() + num_nnz_diag

        assert norm_adj[300, 300].to_dense() == pytest.approx(1)
        assert norm_adj[301, 301].to_dense() == pytest.approx(1)

        for i, j, w in zip(
            norm_adj.storage.row(), norm_adj.storage.col(), norm_adj.storage.value()
        ):
            i = i.item()
            j = j.item()
            denominator = degrees[i if use_in_degrees else j]
            value = (
                0
                if not denominator > 0
                else dir_ms_adj_mat[i, j].to_dense() / denominator
            )
            if i == j and denominator == 0:
                assert norm_adj[i, j].to_dense() == pytest.approx(1)
            else:
                assert norm_adj[i, j].to_dense() == pytest.approx(value)

    @pytest.mark.parametrize("use_in_degrees", [False, True])
    def test_normalized_dense(
        self, dir_ms_adj_mat: tsp.SparseTensor, use_in_degrees: bool
    ):
        """This test assumes that the sparse test above passes."""
        norm_sparse_adj = adj_utils.rw_adj_normalization_tsp(
            dir_ms_adj_mat, use_out_degrees=not use_in_degrees
        )
        norm_dense_adj = adj_utils.rw_adj_normalization_dense(
            dir_ms_adj_mat.to_dense(), use_out_degrees=not use_in_degrees
        )
        assert torch.allclose(norm_dense_adj, norm_sparse_adj.to_dense())


class TestTorchAdj:
    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_instantiation(self, dir_ms_adj_mat: tsp.SparseTensor, dtype: str):
        dtype = torch.float if dtype == "float" else torch.double
        adj_obj = adj_utils.TorchAdj(
            adj=dir_ms_adj_mat, dtype=dtype, remove_self_loops=True
        )
        dir_ms_adj_mat = adj_utils.to_torch_adj(
            dir_ms_adj_mat, dtype=dtype, remove_self_loops=True
        )
        assert adj_obj.adj_ == dir_ms_adj_mat
        assert adj_obj.nnz() == dir_ms_adj_mat.nnz()
        assert adj_obj.num_nodes == dir_ms_adj_mat.size(0)

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_out_norm(self, dir_ms_adj_mat: tsp.SparseTensor, dtype: str):
        dtype = torch.float if dtype == "float" else torch.double
        adj_obj = adj_utils.TorchAdj(
            adj=dir_ms_adj_mat, dtype=dtype, remove_self_loops=True
        )
        assert torch.allclose(
            adj_obj.AD.to_dense(),
            adj_utils.rw_adj_normalization_dense(
                dir_ms_adj_mat.to_dense().to(dtype), use_out_degrees=True
            ),
        )

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_in_norm(self, dir_ms_adj_mat: tsp.SparseTensor, dtype: str):
        dtype = torch.float if dtype == "float" else torch.double
        adj_obj = adj_utils.TorchAdj(
            adj=dir_ms_adj_mat, dtype=dtype, remove_self_loops=True
        )
        assert torch.allclose(
            adj_obj.DA.to_dense(),
            adj_utils.rw_adj_normalization_dense(
                dir_ms_adj_mat.to_dense().to(dtype), use_out_degrees=False
            ),
        )

    @pytest.mark.parametrize("dtype", ["float", "double"])
    def test_sym_norm(self, dir_ms_adj_mat: tsp.SparseTensor, dtype: str):
        dtype = torch.float if dtype == "float" else torch.double
        adj_obj = adj_utils.TorchAdj(
            adj=dir_ms_adj_mat, dtype=dtype, remove_self_loops=True
        )
        answer = adj_utils.to_torch_adj(
            dir_ms_adj_mat.to_dense(), dtype=dtype, remove_self_loops=False
        )
        assert torch.allclose(
            adj_obj.DAD.to_dense(), adj_utils.sym_adj_normalization_dense(answer)
        )

        answer = adj_utils.to_symmetric(answer)
        assert torch.allclose(
            adj_obj.DUD.to_dense(), adj_utils.sym_adj_normalization_dense(answer)
        )

    @pytest.mark.parametrize("free_adj_when_caching", [True, False])
    def test_caching(
        self, dir_ms_adj_mat: tsp.SparseTensor, free_adj_when_caching: bool
    ):
        dtype = torch.float
        dir_ms_adj_mat = dir_ms_adj_mat.to(dtype)
        adj_obj = adj_utils.TorchAdj(
            adj=dir_ms_adj_mat,
            dtype=dtype,
            remove_self_loops=True,
            cache_norm_adj=True,
            free_adj_when_caching=free_adj_when_caching,
        )
        assert adj_obj.adj_ is not None
        assert adj_obj._cached_AD is None
        assert adj_obj._cached_DA is None
        assert adj_obj._cached_DAD is None
        assert adj_obj._cached_UD is None
        assert adj_obj._cached_DUD is None

        ad = adj_obj.AD
        assert adj_obj._cached_AD == ad
        if free_adj_when_caching:
            assert adj_obj.adj_ is None
            with pytest.raises(adj_utils.ClearedCacheError):
                da = adj_obj.DA

        ad2 = adj_obj.AD
        assert ad2 == ad
