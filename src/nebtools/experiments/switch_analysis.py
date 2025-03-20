from typing import List, Dict, Sequence
import datetime
import dataclasses as dc
import pandas as pd
import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import nebtools.data.graph as dgraphs
from nebtools.utils import NEB_DATAROOT
import graph_tool.all as gt
import torch
import nebtools.algs.preconfigs as embalgsets
import nebtools.algs.utils as algutils
import nebtools.experiments.classification as nodeclassification
import nebtools.experiments.pt_sgd_log_reg as sgd_classification

import reachnes.adj_utils as rn_adjutils
import reachnes.ew_filtering as rn_filter
import reachnes.reachability as rability
import reachnes.coeffs as rcoeffs
import nebtools.data.core_ as datacore


@dc.dataclass
class EvalModelData:
    lr_model: sgd_classification.LogisticRegression
    embeddings: np.ndarray
    labels: np.ndarray
    node_index_train: np.ndarray
    node_index_test: np.ndarray


def load_graph(dataset: str):
    dataroot = NEB_DATAROOT
    dataset_spec = dgraphs.DatasetSpec(
        data_name=dataset,
        force_undirected=False,
        force_unweighted=True,
        rm_node_attributes=False,
        with_self_loops=False,
    )
    data_graph = dgraphs.SimpleGraph.from_dataset_spec(
        dataroot=dataroot, dataset_spec=dataset_spec
    )
    return data_graph


def compute_reachability(
    graph,
    nodes,
    rw_distribution,
    tau,
    loc,
    adj_seq,
    use_log_transform: bool = False,
    device=None,
    order: int = 10,
):
    dtype = torch.float32
    if device is None:
        device = torch.device("cuda")
    adj = datacore.edges2spmat(
        graph.edges, graph.weights, num_nodes=graph.num_nodes, directed=graph.directed
    )
    adj_obj = rn_adjutils.TorchAdj(adj=adj, dtype=dtype, remove_self_loops=False).to(
        device
    )
    coeff_spec = rcoeffs.CoeffsSpec(name=rw_distribution, kwargs={"tau": tau}, loc=loc)
    coeffs_obj = rcoeffs.RWLCoefficientsModel.from_rwl_distributions(
        [coeff_spec], order=order, dtype=dtype, device=device, normalize=True
    )
    coeffs = coeffs_obj()
    adj_seq = rn_adjutils.AdjSeq(seq=adj_seq)
    batch = torch.tensor(nodes, dtype=torch.long, device=device)
    reachability = rability.compute_reachability(
        adj_obj=adj_obj, batch_indices=batch, coeffs=coeffs, adj_seq=adj_seq
    )
    if use_log_transform:
        filter = rn_filter.LogFilter(scaling_factor=graph.num_nodes)
        reachability = filter(reachability)

    return reachability[0, :, :].cpu()


def gini_torch(x: torch.Tensor):
    sorted_x = torch.sort(x, dim=0)[0]
    n = x.shape[0]
    cumx = torch.cumsum(sorted_x, dim=0, dtype=torch.float32)
    return (n + 1 - 2 * torch.sum(cumx, dim=0) / cumx[-1, :]) / n


def batch_sampler_generator(num_nodes, batch_size, device):
    for i in range(0, num_nodes, batch_size):
        yield torch.arange(
            i, min(i + batch_size, num_nodes), dtype=torch.long, device=device
        )


def random_batch_sampler_generator(num_nodes, batch_size, num_samples, device):
    random_nodes = torch.randperm(num_nodes, dtype=torch.long, device=device)[
        :num_samples
    ]
    for i in range(0, num_samples, batch_size):
        yield random_nodes[i : i + batch_size]


def selected_nodes_batch_sampler_generator(
    selected_nodes: torch.LongTensor, batch_size, device
):
    selected_nodes = selected_nodes.to(device)
    for i in range(0, len(selected_nodes), batch_size):
        yield selected_nodes[i : i + batch_size]


def get_batch_sampler(
    *,
    num_nodes: int,
    batch_size: int,
    num_samples: int,
    device,
    selected_nodes: torch.LongTensor = None,
):
    if selected_nodes is not None:
        batch_sampler = selected_nodes_batch_sampler_generator(
            selected_nodes, batch_size=batch_size, device=device
        )
        num_batches = int(np.ceil(len(selected_nodes) / batch_size))
    elif num_samples < 0 or num_samples >= num_nodes:
        batch_sampler = batch_sampler_generator(
            num_nodes=num_nodes, batch_size=batch_size, device=device
        )
        num_batches = int(np.ceil(num_nodes / batch_size))
    else:
        batch_sampler = random_batch_sampler_generator(
            num_nodes=num_nodes,
            batch_size=batch_size,
            device=device,
            num_samples=num_samples,
        )
        num_batches = int(np.ceil(min(num_samples, num_nodes) / batch_size))
    return batch_sampler, num_batches


def compute_dispersal(
    graph,
    adj_seqs,
    coeff_spec,
    *,
    batch_size: int = 2048,
    num_samples: int = -1,
    num_steps: int,
    selected_nodes: torch.LongTensor = None,
):
    coverage_results = []
    entropy_results = []

    dtype = torch.float32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    adj = datacore.edges2spmat(
        graph.edges, graph.weights, num_nodes=graph.num_nodes, directed=graph.directed
    )
    adj_obj = rn_adjutils.TorchAdj(adj=adj, dtype=dtype, remove_self_loops=False).to(
        device
    )
    coeffs_obj = rcoeffs.RWLCoefficientsModel.from_rwl_distributions(
        [coeff_spec], order=num_steps, dtype=dtype, device=device, normalize=True
    )
    coeffs = coeffs_obj()
    adj_seqs = adj_seqs.split("::")

    batch_sampler, num_batches = get_batch_sampler(
        num_nodes=graph.num_nodes,
        batch_size=batch_size,
        device=device,
        num_samples=num_samples,
        selected_nodes=selected_nodes,
    )

    for i, batch in enumerate(batch_sampler):
        # batch = torch.arange(i, min(i + batch_size, graph.num_nodes), dtype=torch.long, device=device)
        if i % 100 == 0 and num_batches > 100:
            print(f"{datetime.datetime.now().isoformat()} -- {i} / {num_batches}")
        reachability = torch.zeros(
            1, graph.num_nodes, len(batch), device=device, dtype=dtype
        )
        for adj_seq in adj_seqs:
            adj_seq = rn_adjutils.AdjSeq(seq=adj_seq)

            reachability += rability.compute_reachability(
                adj_obj=adj_obj, batch_indices=batch, coeffs=coeffs, adj_seq=adj_seq
            )
        reachability = reachability / len(adj_seqs)
        coverage = compute_coverage(reachability[0], threshold=1.0 / graph.num_nodes)
        entropy = compute_entropy(reachability[0])
        coverage_results.append(coverage)
        entropy_results.append(entropy)
    coverage_results = torch.cat(coverage_results).cpu().numpy()
    entropy_results = torch.cat(entropy_results).cpu().numpy()
    return coverage_results, entropy_results


def compute_entropy(x):
    return -(x * torch.log2(x + 1e-10)).sum(dim=0)


def compute_coverage(x, threshold: float):
    return (x >= threshold).sum(dim=0)


def compute_total_variation_distance(reachability: torch.Tensor):
    reachability = reachability.permute(2, 0, 1)
    tv_distance = 0.5 * torch.cdist(reachability, reachability, p=1)
    return tv_distance


def compute_hellinger_distance(reachability: torch.Tensor):
    sqr_reachability = torch.sqrt(reachability.permute(2, 0, 1))
    bc_coeff = torch.einsum("...ik,...jk->...ij", sqr_reachability, sqr_reachability)
    hellinger_dist = torch.sqrt(1.0 - bc_coeff)
    return hellinger_dist


def compute_stat_distances_for_all_nodes(
    graph, adj_seqs, coeff_specs, batch_size: int = 2048, *, num_samples: int = -1
):
    hellinger_distances = []
    tv_distances = []
    dtype = torch.float32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    adj = datacore.edges2spmat(
        graph.edges, graph.weights, num_nodes=graph.num_nodes, directed=graph.directed
    )
    adj_obj = rn_adjutils.TorchAdj(adj=adj, dtype=dtype, remove_self_loops=False).to(
        device
    )
    coeffs_obj = rcoeffs.RWLCoefficientsModel.from_rwl_distributions(
        coeff_specs, order=12, dtype=dtype, device=device, normalize=True
    )
    coeffs = coeffs_obj()

    batch_sampler, num_batches = get_batch_sampler(
        num_nodes=graph.num_nodes,
        batch_size=batch_size,
        device=device,
        num_samples=num_samples,
    )

    for i, batch in enumerate(batch_sampler):
        # batch = torch.arange(i, min(i + batch_size, graph.num_nodes), dtype=torch.long, device=device)
        if i % 100 == 0 and num_batches > 100:
            print(f"{datetime.datetime.now().isoformat()} -- {i} / {num_batches}")
        reachability = []
        for adj_seq in adj_seqs.split("::"):
            adj_seq = rn_adjutils.AdjSeq(seq=adj_seq)
            reachability.append(
                rability.compute_reachability(
                    adj_obj=adj_obj, batch_indices=batch, coeffs=coeffs, adj_seq=adj_seq
                )
            )
        reachability = torch.cat(reachability, dim=0)
        tv_dist = compute_total_variation_distance(reachability)
        tv_distances.append(tv_dist)
    tv_distances = torch.cat(tv_distances, dim=0)
    return hellinger_distances, tv_distances


if __name__ == "__main__":
    edges = np.asarray(
        [
            [0, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [2, 4],
            # [5, 4]
        ]
    )
    pos = np.asarray([[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]])
    num_nodes = edges.max() + 1
    graph = dgraphs.SimpleGraph(num_nodes=num_nodes, edges=edges, directed=True)
    reachabilityU = compute_reachability(graph, list(range(5)), "poisson", 1, 1, "U")
