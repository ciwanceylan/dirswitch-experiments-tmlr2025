import pytest
import numpy as np
import scipy
from scipy.special import loggamma
import torch

import reachnes.reachability as rn_ra
import reachnes.coeffs as rn_coeffs
import reachnes.source_star_graph as rn_ssg
import reachnes.adj_utils as adj_utils


def compare_taylor_coeff(scale, order):
    log_val = -scale + np.log(scale) * np.arange(order) - loggamma(np.arange(order) + 1)
    return np.power(-1, np.arange(order)) * np.exp(log_val)


def gamma_k_n(n, K, dtype=np.float64):
    """Round-off error term from Higham"""
    u = np.finfo(dtype).eps
    nku = n * K * u
    return nku / (1 - nku)


def one_taylor_bound(n, K, tau, dtype=np.float64):
    """Taylor approximation error bound for matrix exponential"""
    log_approx_bound = (K + 1) * np.log(tau) + tau - loggamma(K + 2)
    approx_bound = np.exp(log_approx_bound)
    bound = (gamma_k_n(n, K, dtype=dtype) + approx_bound) * np.exp(tau)
    return bound


def source_start_adj(directed: bool):
    adj = rn_ssg.source_star_adj(6, 3, 2, directed=directed)
    adj = adj_utils.to_torch_adj(adj, dtype=torch.double, remove_self_loops=False)
    return adj


def diamond_adj(directed: bool):
    adj = rn_ssg.diamond_adj(50, directed=directed)
    adj = adj_utils.to_torch_adj(adj, dtype=torch.double, remove_self_loops=False)
    return adj


def cycle_with_chords_adj(directed: bool):
    adj = rn_ssg.cycle_with_chords_adj(20, num_chords=2, directed=directed)
    adj = adj_utils.to_torch_adj(adj, dtype=torch.double, remove_self_loops=False)
    return adj


def adjacency_matrix_normalization(adj, seq: str):
    out_degrees = adj.sum(dim=0)
    inv_out_deg = torch.zeros_like(out_degrees)
    inv_out_deg[out_degrees > 0] = 1.0 / out_degrees[out_degrees > 0]

    in_degrees = adj.sum(dim=1)
    inv_in_deg = torch.zeros_like(in_degrees)
    inv_in_deg[in_degrees > 0] = 1.0 / in_degrees[in_degrees > 0]

    undir_adj = adj + adj.T
    degrees = undir_adj.sum(dim=0)
    inv_deg = torch.zeros_like(degrees)
    inv_deg[degrees > 0] = 1.0 / degrees[degrees > 0]

    if seq == "O":
        norm_adj = adj * inv_out_deg.view(1, -1)
        norm_adj += torch.diag(out_degrees < 1e-6)
    elif seq == "I":
        norm_adj = (adj * inv_in_deg.view(-1, 1)).T
        norm_adj += torch.diag(in_degrees < 1e-6)
    elif seq == "F":
        norm_adj = adj * inv_in_deg.view(-1, 1)
        norm_adj += torch.diag(in_degrees < 1e-6)
    elif seq == "B":
        norm_adj = (adj * inv_out_deg.view(1, -1)).T
        norm_adj += torch.diag(out_degrees < 1e-6)
    elif seq == "X":
        norm_adj = (
            torch.sqrt(inv_in_deg.view(-1, 1))
            * adj
            * torch.sqrt(inv_out_deg.view(1, -1))
        )
        norm_adj += torch.diag((out_degrees + in_degrees) < 1e-6)
    elif seq == "S":
        norm_adj = (
            torch.sqrt(inv_deg.view(-1, 1))
            * undir_adj
            * torch.sqrt(inv_deg.view(1, -1))
        )
        norm_adj += torch.diag(degrees < 1e-6)
    elif seq == "U":
        norm_adj = undir_adj * inv_deg.view(1, -1)
        norm_adj += torch.diag(degrees < 1e-6)
    elif seq == "C":
        norm_adj = undir_adj * inv_deg.view(-1, 1)
        norm_adj += torch.diag(degrees < 1e-6)
    elif seq == "A":
        norm_adj = adj
    elif seq == "T":
        norm_adj = adj.T
    else:
        raise NotImplementedError(f"Orientation '{seq}' not implemented.")
    return norm_adj


@pytest.mark.parametrize("graph", ["ss", "diamond", "cwc"])
@pytest.mark.parametrize("as_dense", [True, False])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seq", ["O", "I", "S", "A", "T", "F", "B", "X", "U", "C"])
def test_adj_matmul(graph: str, as_dense: bool, directed: bool, seq: str):
    dtype = torch.float
    if graph == "ss":
        adj = source_start_adj(directed=directed)
    elif graph == "diamond":
        adj = diamond_adj(directed=directed)
    elif graph == "cwc":
        adj = cycle_with_chords_adj(directed=directed)
    else:
        raise NotImplementedError(f"Graph '{graph}' not implemented.")

    adj = adj.to(dtype=dtype)
    x = torch.randn(adj.size(0), 20, dtype=dtype)
    if as_dense:
        adj = adj.to_dense()

    adj_obj = adj_utils.TorchAdj(
        adj=adj, dtype=dtype, cache_norm_adj=True, remove_self_loops=True
    )

    answer = adjacency_matrix_normalization(adj.to_dense(), seq) @ x
    result = rn_ra.adj_matmul(adj_obj, x, orientation=seq)
    torch.testing.assert_close(result, answer)


@pytest.mark.parametrize("graph", ["ss", "diamond", "cwc"])
@pytest.mark.parametrize("as_dense", [False, True])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seq", ["O", "I", "S", "A", "T", "F", "B", "X", "U", "C"])
@pytest.mark.parametrize("dtype", ["float", "double"])
def test_expm(graph: str, as_dense: bool, directed: bool, seq: str, dtype: str):
    np_dtype = np.float32 if dtype == "float" else np.float64
    dtype = torch.float if dtype == "float" else torch.double
    if graph == "ss":
        adj = source_start_adj(directed=directed)
    elif graph == "diamond":
        adj = diamond_adj(directed=directed)
    elif graph == "cwc":
        adj = cycle_with_chords_adj(directed=directed)
    else:
        raise NotImplementedError(f"Graph '{graph}' not implemented.")

    if as_dense:
        adj = adj.to_dense()

    adj_obj = adj_utils.TorchAdj(
        adj=adj, dtype=dtype, cache_norm_adj=True, remove_self_loops=True
    )
    if seq == "O":
        scipy_norm_adj = adj_obj.AD.to_dense().numpy()
    elif seq == "I":
        scipy_norm_adj = adj_obj.DA.t().to_dense().numpy()
    elif seq == "F":
        scipy_norm_adj = adj_obj.DA.to_dense().numpy()
    elif seq == "B":
        scipy_norm_adj = adj_obj.AD.t().to_dense().numpy()
    elif seq == "X":
        scipy_norm_adj = adj_obj.DAD.to_dense().numpy()
    elif seq == "S":
        scipy_norm_adj = adj_obj.DUD.to_dense().numpy()
    elif seq == "U":
        scipy_norm_adj = adj_obj.UD.to_dense().numpy()
    elif seq == "C":
        scipy_norm_adj = adj_obj.UD.t().to_dense().numpy()
    elif seq == "A":
        scipy_norm_adj = adj_obj.adj_.to_dense().numpy()
    elif seq == "T":
        scipy_norm_adj = adj_obj.adj_.t().to_dense().numpy()
    else:
        raise NotImplementedError(f"Orientation '{seq}' not implemented.")
    adj_seq = adj_utils.AdjSeq(seq)
    taus = (1.0, 6.0)

    coeff_specs = [
        rn_coeffs.CoeffsSpec(name="poisson", kwargs={"tau": tau}, loc=0) for tau in taus
    ]
    coeffs_obj = rn_coeffs.RWLCoefficientsModel.from_rwl_distributions(
        coeff_specs, order=30, dtype=dtype, device=torch.device("cpu"), normalize=True
    )

    reachability_obj = rn_ra.ReachabilityModel(coeffs_obj=coeffs_obj)
    reachability = reachability_obj(
        adj_obj=adj_obj,
        adj_seq=adj_seq,
        batch_indices=torch.arange(adj_obj.num_nodes).to(torch.long),
    )

    for k_tau, tau in enumerate(taus):
        if seq in ["A", "T"]:
            # Test is numerically inaccurate without normalization
            continue
        scipy_res = np.exp(-tau) * scipy.linalg.expm(tau * scipy_norm_adj)
        error = torch.linalg.matrix_norm(
            torch.from_numpy(scipy_res) - reachability[k_tau], ord=1
        )
        assert error < one_taylor_bound(adj_obj.num_nodes, 30 - 1, tau, dtype=np_dtype)


@pytest.mark.parametrize("dtype", ["float", "double"])
def test_alternating_sequences(dtype):
    dtype = torch.float if dtype == "float" else torch.double
    adj = cycle_with_chords_adj(directed=False)
    adj_obj = adj_utils.TorchAdj(adj=adj, dtype=dtype, remove_self_loops=True)
    adj_seq = adj_utils.AdjSeq("OIS")

    coeff_specs = [
        rn_coeffs.CoeffsSpec(name="geometric", kwargs={"alpha": 0.9}, loc=0),
        rn_coeffs.CoeffsSpec(name="geometric", kwargs={"alpha": 0.5}, loc=0),
    ]
    coeffs_obj = rn_coeffs.RWLCoefficientsModel.from_rwl_distributions(
        coeff_specs, order=3, dtype=dtype, device=torch.device("cpu"), normalize=True
    )

    reachability_obj = rn_ra.ReachabilityModel(coeffs_obj=coeffs_obj)
    reachability = reachability_obj(
        adj_obj=adj_obj,
        adj_seq=adj_seq,
        batch_indices=torch.arange(adj_obj.num_nodes).to(torch.long),
    )

    eye = torch.eye(adj_obj.num_nodes, dtype=dtype)
    out_norm = adj_obj.AD.to_dense()
    in_norm = adj_obj.DA.t().to_dense()
    sym_norm = adj_obj.DAD.to_dense()
    coeffs = coeffs_obj()
    for s in range(2):
        answer = (
            coeffs[s][0] * eye
            + coeffs[s][1] * out_norm
            + coeffs[s][2] * (in_norm @ out_norm)
            + coeffs[s][3] * (sym_norm @ in_norm @ out_norm)
        )
        torch.testing.assert_close(reachability[s], answer.to_dense())


@pytest.mark.parametrize("graph", ["ss", "diamond", "cwc"])
@pytest.mark.parametrize("as_dense", [False, True])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("seq", ["O", "I", "S", "F", "B"])
@pytest.mark.parametrize("dtype", ["float", "double"])
def test_reachnes_with_x(
    graph: str, as_dense: bool, directed: bool, seq: str, dtype: str
):
    dtype = torch.float if dtype == "float" else torch.double
    if graph == "ss":
        adj = source_start_adj(directed=directed)
    elif graph == "diamond":
        adj = diamond_adj(directed=directed)
    elif graph == "cwc":
        adj = cycle_with_chords_adj(directed=directed)
    else:
        raise NotImplementedError(f"Graph '{graph}' not implemented.")

    adj_obj = adj_utils.TorchAdj(
        adj=adj, dtype=dtype, cache_norm_adj=True, remove_self_loops=True
    )
    x = torch.rand(size=(adj_obj.num_nodes, 25), dtype=dtype)
    adj_seq = adj_utils.AdjSeq(seq)
    taus = (1.0, 6.0)

    coeff_specs = [
        rn_coeffs.CoeffsSpec(name="poisson", kwargs={"tau": tau}, loc=0) for tau in taus
    ]
    coeffs_obj = rn_coeffs.RWLCoefficientsModel.from_rwl_distributions(
        coeff_specs, order=30, dtype=dtype, device=torch.device("cpu"), normalize=True
    )

    reachability_obj = rn_ra.ReachabilityModel(coeffs_obj=coeffs_obj)
    reachability = reachability_obj(
        adj_obj=adj_obj,
        adj_seq=adj_seq,
        batch_indices=torch.arange(adj_obj.num_nodes).to(torch.long),
    )
    answer = reachability @ x

    reachability_with_x_obj = rn_ra.ReachabilityTimesXModel(coeffs_obj=coeffs_obj)
    reachability_x = reachability_with_x_obj(adj_obj=adj_obj, adj_seq=adj_seq, x=x)
    torch.testing.assert_close(reachability_x, answer)
