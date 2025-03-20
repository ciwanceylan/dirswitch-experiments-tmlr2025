from typing import Tuple
import pytest
import torch
import torch_sparse as tsp
import reachnes.coeffs as rn_coeffs
import reachnes.reachability as rn_ra
import reachnes.reduction as rn_reduc
import reachnes.adj_utils as adj_utils


@pytest.fixture()
def fake_reachability(dir_ms_adj_mat):
    adj_obj = adj_utils.TorchAdj(
        adj=dir_ms_adj_mat,
        dtype=torch.float,
        cache_norm_adj=True,
        remove_self_loops=True,
    )
    adj_seq = adj_utils.AdjSeq("O")
    batch_indices = torch.arange(dir_ms_adj_mat.size(0), dtype=torch.long).to(
        torch.long
    )
    coeffs = rn_coeffs.poisson_coefficients(
        tau=2, num_coeffs=4, dtype=torch.float, device=torch.device("cpu")
    ).view(1, -1)
    reachability = rn_ra.compute_reachability(
        adj_obj=adj_obj,
        batch_indices=batch_indices,
        coeffs=coeffs,
        adj_seq=adj_seq,
    )
    out = torch.cat(3 * [reachability], dim=0)
    return out


@pytest.fixture()
def fake_reachability_sparse(fake_reachability) -> Tuple[tsp.SparseTensor]:
    out = tuple(tsp.SparseTensor.from_dense(mat) for mat in fake_reachability)
    return out


def ecf_check(
    reachability: torch.Tensor, time_points: torch.Tensor, use_energy_distance: bool
):
    num_nodes = reachability.shape[0]
    tmp = reachability.unsqueeze(-1) * time_points.view(1, 1, -1)
    cos_vals = torch.cos(tmp).sum(dim=0) / num_nodes
    sin_vals = torch.sin(tmp).sum(dim=0) / num_nodes
    embeddings = []
    if use_energy_distance:
        for i, t in enumerate(time_points):
            embeddings.append(cos_vals[:, i] / t)
            embeddings.append(sin_vals[:, i] / t)
    else:
        for i, t in enumerate(time_points):
            embeddings.append(cos_vals[:, i])
            embeddings.append(sin_vals[:, i])
    return torch.stack(embeddings, dim=1)


def ecf_model(num_series: int, use_energy_distance: bool):
    model = rn_reduc.ECFReduction(use_energy_distance=use_energy_distance)
    model = model.init(emb_dim=8, num_series=num_series, num_nodes=-1)
    return model


def moments_model(num_series: int):
    model = rn_reduc.CentralMomentsReduction()
    model = model.init(emb_dim=8, num_series=num_series, num_nodes=-1)
    return model


def sorted_values_model(num_series: int):
    model = rn_reduc.SortedValuesReduction()
    model = model.init(emb_dim=8, num_series=num_series, num_nodes=-1)
    return model


def spr_svd_proximal_model(num_series: int, num_nodes: int):
    model = rn_reduc.SPRSVDProximalReduction(
        num_oversampling=8, block_size=2, include_v=True
    )
    model = model.init(emb_dim=8, num_nodes=num_nodes, num_series=num_series)
    return model


# def spr_svd_strucural_model(num_series: int, num_nodes: int):
#     return rn_reduc.SPRSVDStructuralReduction(num_nodes=num_nodes, k=8, num_oversampling=8, num_series=num_series,
#                                               block_size=2)


def svd_proximal_model(num_series: int):
    model = rn_reduc.SVDProximalReduction(include_v=True)
    model = model.init(emb_dim=8, num_series=num_series, num_nodes=-1)
    return model


# def svd_structural_model(num_series: int):
#     return rn_reduc.SVDStructuralReduction(k=8, num_series=num_series)


@pytest.mark.parametrize(
    "model_name", ["ecf", "moments", "spr_svd_prox", "svd_prox", "sorted_values"]
)
def test_reduction_flow(fake_reachability: torch.Tensor, model_name: str):
    num_series = fake_reachability.shape[0]
    num_total_nodes = fake_reachability.shape[1]
    num_embeddings = fake_reachability.shape[2]
    dtype = fake_reachability.dtype

    if model_name == "ecf":
        model = ecf_model(num_series, use_energy_distance=True)
    elif model_name == "moments":
        model = moments_model(num_series)
    elif model_name == "spr_svd_prox":
        model = spr_svd_proximal_model(num_series, num_total_nodes)
    # elif model_name == "spr_svd_struct":
    #     model = spr_svd_strucural_model(num_series, num_total_nodes)
    elif model_name == "svd_prox":
        model = svd_proximal_model(num_series)
    # elif model_name == "svd_struct":
    #     model = svd_structural_model(num_series)
    elif model_name == "sorted_values":
        model = sorted_values_model(num_series)
    else:
        raise NotImplementedError(f"Model '{model_name}' not implemented")

    model = model.to(dtype=dtype, device=torch.device("cpu"))
    batch_size = 24

    gathered_pre_embeddings = rn_reduc.GatheredPreEmbeddings()
    for i in range(0, num_embeddings, batch_size):
        start = i
        end = min(i + batch_size, num_embeddings)
        batch_indices = torch.arange(i, end, dtype=torch.long).to(torch.long)
        batch = fake_reachability[:, :, start:end]
        pre_embeddings_batch = model.reachability2pre_embeddings(batch, batch_indices)
        gathered_pre_embeddings.store_pre_embeddings(pre_embeddings_batch)

    pre_embeddings = model.reduce_gathered_pre_embeddings(
        gathered_pre_embeddings=gathered_pre_embeddings
    )
    embeddings = model(pre_embeddings)
    assert embeddings.shape[0] == num_series
    assert embeddings.shape[1] == num_embeddings
    assert embeddings.shape[2] == 8

    if model_name == "ecf":
        ecf_t = (
            torch.exp(model.log_max_val)
            * torch.arange(
                1, model.num_eval_points + 1, step=1, device=torch.device("cpu")
            )
            / model.num_eval_points
        )
        control_embeddings = rn_reduc.ecf_dense(
            fake_reachability, ecf_t, use_energy_distance=True
        )
        assert embeddings.dtype == control_embeddings.dtype
        torch.testing.assert_close(embeddings, control_embeddings)
    elif model_name == "moments":
        control_embeddings = rn_reduc.compute_central_moments_dense(
            fake_reachability, num_moments=model.num_moments
        )
        assert embeddings.dtype == control_embeddings.dtype
        torch.testing.assert_close(embeddings, control_embeddings, rtol=3e-5, atol=3e-5)
    elif model_name == "spr_svd_prox":
        for s in range(num_series):
            G = fake_reachability[s].t() @ model.omega[s]
            H = fake_reachability[s] @ G
            torch.testing.assert_close(G, pre_embeddings.data[f"G_{s}"])
            torch.testing.assert_close(
                H, pre_embeddings.data[f"H_{s}"], rtol=3e-5, atol=3e-5
            )
    elif model_name == "sorted_values":
        fake_reach_diagonal = torch.diagonal(fake_reachability, dim1=1, dim2=2)
        torch.testing.assert_close(embeddings[:, :, 0], fake_reach_diagonal)
        new_fake_reachability = torch.diagonal_scatter(
            fake_reachability,
            src=torch.full_like(fake_reach_diagonal, fill_value=-1.0),
            dim1=1,
            dim2=2,
        )
        sorted_fake_reachability, _ = torch.sort(
            new_fake_reachability, dim=1, descending=True
        )
        emb_gt_values = sorted_fake_reachability[:, :7, :].transpose(1, 2)
        torch.testing.assert_close(embeddings[:, :, 1:], emb_gt_values)

    # elif model_name == "spr_svd_struct":
    #     for s in range(num_series):
    #         reacha, _ = torch.sort(fake_reachability[s], descending=True, dim=0)
    #         G = reacha.t() @ model.omega[s]
    #         H = reacha @ G
    #         torch.testing.assert_close(G, pre_embeddings.data[f"G_{s}"])
    #         torch.testing.assert_close(H, pre_embeddings.data[f"H_{s}"], rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("dtype", ["float", "double"])
def test_svd_proximal(
    fake_reachability,
    fake_reachability_sparse: tsp.SparseTensor,
    sparse_input: bool,
    dtype,
):
    dtype = torch.float if dtype == "float" else torch.double
    fake_reachability = fake_reachability[[0]].to(dtype=dtype)
    if sparse_input:
        reachability_input = (fake_reachability_sparse[0].to(dtype=dtype),)
    else:
        reachability_input = fake_reachability
    num_series = fake_reachability.shape[0]
    num_nodes = fake_reachability.shape[1]
    num_embeddings = fake_reachability.shape[2]
    assert num_series == 1

    # dtype = torch.float if dtype == "float" else torch.double
    # fake_reachability = fake_reachability.to(dtype=dtype)
    model = rn_reduc.SVDProximalReduction(include_v=True)
    model = model.init(
        emb_dim=2 * num_nodes, num_series=num_series, num_nodes=num_nodes
    ).to(dtype=dtype)

    batch_indices = torch.arange(0, num_embeddings, dtype=torch.long).to(torch.long)
    pre_embeddings = model.reachability2pre_embeddings(
        reachability_input, batch_indices
    )
    embeddings = model(pre_embeddings)
    u, v = torch.chunk(embeddings[0], chunks=2, dim=1)
    recon = v @ u.t()  # Note the transposition
    torch.testing.assert_close(recon, fake_reachability[0])


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("dtype", ["float", "double"])
def test_sprsvd_proximal(
    fake_reachability: torch.Tensor,
    fake_reachability_sparse: tsp.SparseTensor,
    sparse_input: bool,
    dtype: str,
):
    dtype = torch.float if dtype == "float" else torch.double
    fake_reachability = fake_reachability[[0]].to(dtype=dtype)
    if sparse_input:
        reachability_input = (fake_reachability_sparse[0].to(dtype=dtype),)
    else:
        reachability_input = fake_reachability
    num_series = fake_reachability.shape[0]
    num_nodes = fake_reachability.shape[1]
    num_embeddings = fake_reachability.shape[2]
    assert num_series == 1

    model = rn_reduc.SPRSVDProximalReduction(
        num_oversampling=0, block_size=1, include_v=True
    )
    model = model.init(
        emb_dim=2 * num_nodes, num_series=num_series, num_nodes=num_nodes
    ).to(dtype=dtype)

    batch_indices = torch.arange(0, num_embeddings, dtype=torch.long).to(torch.long)
    pre_embeddings = model.reachability2pre_embeddings(
        reachability_input, batch_indices
    )
    embeddings = model(pre_embeddings)
    u, v = torch.chunk(embeddings[0], chunks=2, dim=1)
    recon = v @ u.t()  # Note the transposition
    if dtype == torch.float:
        torch.testing.assert_close(recon, fake_reachability[0], atol=5e-4, rtol=1e-4)
    else:
        torch.testing.assert_close(recon, fake_reachability[0])


@pytest.mark.parametrize("max_val", [0.1, 0.5, 1, 3, 50, 100, 5000])
@pytest.mark.parametrize("use_energy_distance", [True, False])
def test_ecf_dense(fake_reachability, max_val, use_energy_distance):
    num_series = fake_reachability.shape[0]
    num_eval_points = 10
    ecf_t = (
        max_val
        * torch.arange(1, num_eval_points + 1, step=1, device=torch.device("cpu"))
        / num_eval_points
    )
    res = rn_reduc.ecf_dense(fake_reachability, ecf_t, use_energy_distance)
    for s in range(num_series):
        answer = ecf_check(
            fake_reachability[s], ecf_t, use_energy_distance=use_energy_distance
        )
        torch.testing.assert_close(res[s], answer)


@pytest.mark.parametrize("use_energy_distance", [True, False])
def test_ecf_sparse(fake_reachability, fake_reachability_sparse, use_energy_distance):
    num_eval_points = 10
    ecf_t = torch.arange(1, num_eval_points + 1, step=1, device=torch.device("cpu"))
    ecf_sp = rn_reduc.ecf_multi_sparse(
        fake_reachability_sparse, ecf_t, use_energy_distance=use_energy_distance
    )
    ecf_dense = rn_reduc.ecf_dense(
        fake_reachability, ecf_t, use_energy_distance=use_energy_distance
    )

    for s in range(fake_reachability.shape[0]):
        torch.testing.assert_close(ecf_sp.to_dense(), ecf_dense)
