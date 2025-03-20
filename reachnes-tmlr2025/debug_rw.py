import numpy as np
import networkx as nx
import torch
import reachnes.coeffs as rn_coeffs
import reachnes.reachnes_random_walks as rn_rw


def main():
    torch.autograd.set_detect_anomaly(True)
    # device = torch.device('cuda')

    as_undirected = True
    # graph = nx.cycle_graph(100, create_using=nx.DiGraph)
    graph = nx.read_edgelist(
        "./datasets/arenas_clean.edgelist", comments="%", delimiter=" "
    )
    if as_undirected:
        graph = graph.to_undirected()
    adj = nx.adjacency_matrix(graph).tocoo()
    edge_index = torch.from_numpy(np.stack((adj.row, adj.col), axis=0))

    coeffs_obj = rn_coeffs.FixedReachabilityCoefficients.from_spec(
        spec=rn_coeffs.CoeffsSpec(name="uniform", kwargs={"nnz": 10}, prop_zero=None),
        num_coeffs=10,
        dtype=torch.float,
        device=torch.device("cpu"),
        normalize=False,
    )
    params = rn_rw.RNRWParams(
        num_nodes=adj.shape[0],
        sub_embedding_dim=64,
        use_forward_edges=True,
        use_reverse_edges=False,
        walks_per_node=10,
    )

    trainer = rn_rw.RNRWTrainer(
        edge_index=edge_index,
        params=params,
        coeffs_obj=coeffs_obj,
        device=torch.device("cpu"),
    )

    trainer.run_training(10)


if __name__ == "__main__":
    main()
