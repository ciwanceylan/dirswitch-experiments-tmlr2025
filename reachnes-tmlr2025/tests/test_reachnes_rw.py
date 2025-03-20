import torch
import reachnes.coeffs as rn_coeffs
import reachnes.run_reachnes as rn_run
from .test_reachability import source_start_adj


def test_reachnes_rw():
    order = 10
    num_epochs = 3
    sampled_walk_length = 20
    walks_per_node = 1
    num_negative_samples = 3
    sparse = True
    batch_size = 512
    coeffs_specs = (rn_coeffs.CoeffsSpec(name="geometric", kwargs={"tau": 1.0}, loc=0),)
    rw_spec = rn_run.ReachnesRWSpecification(
        emb_dim=128,
        order=order,
        coeffs=coeffs_specs,
        use_forward_edges=True,
        use_reverse_edges=True,
        num_epochs=num_epochs,
        sampled_walk_length=sampled_walk_length,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        sparse=sparse,
        cpu_workers=4,
        batch_size=batch_size,
        lr=0.01,
        use_float64=False,
        use_cpu=True,
        as_src_dst_tuple=False,
    )

    adj = source_start_adj(directed=True)
    edge_index = torch.stack((adj.storage.row(), adj.storage.col()), dim=0)
    num_nodes = adj.size(0)
    embeddings = rn_run.run_rn_rw(
        edge_index=edge_index, num_nodes=num_nodes, rn_spec=rw_spec
    )
