import dataclasses as dc
from typing import Sequence, Literal, Tuple, Optional

import torch
import torch_sparse as tsp

import nebtools.data.graph as dgraph
from nebtools.ssgnns.utils import SSGNNTrainer

from .BLADE.model import BLADENet
from .BLADE.loss import BLADELoss


@dc.dataclass(frozen=True)
class Parameters:
    num_epochs: int = 30
    lr: float = 1e-4
    emb_dim: int = 128
    num_layers: int = 3
    neg_per_pos: int = 1
    use_pos_edge_score: bool = True
    init_method: Literal["normal"] = "normal"


class BladeTrainer(SSGNNTrainer):
    def __init__(
        self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device
    ):
        self.model = BLADENet(
            in_dim=params.emb_dim, out_dim=params.emb_dim, num_layers=params.num_layers
        )
        self.blade_loss = BLADELoss(
            neg_per_pos=params.neg_per_pos, use_pos_edge_score=params.use_pos_edge_score
        )

        self.init_x = self.model.initialize_features(
            method=params.init_method, num_nodes=graph.num_nodes
        ).to(device)
        edge_index = torch.from_numpy(graph.edges).t().to(device)
        self.adj = tsp.SparseTensor.from_edge_index(
            edge_index=edge_index,
            sparse_sizes=(graph.num_nodes, graph.num_nodes),
        )
        self.model.to(device)

        self.blade_loss.set_positive_edge_index(edge_index, num_nodes=graph.num_nodes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.device = device

    def step(self, step: int):
        self.model.train()

        emb_s, emb_t = self.model(x=self.init_x, adj=self.adj)
        loss = self.blade_loss.loss(emb_s=emb_s, emb_t=emb_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_embeddings(self) -> torch.Tensor:
        self.model.eval()
        embeds_s, embeds_t = self.model.embed(x=self.init_x, adj=self.adj)
        embeds = torch.concat((embeds_s, embeds_t), dim=1)
        return embeds
