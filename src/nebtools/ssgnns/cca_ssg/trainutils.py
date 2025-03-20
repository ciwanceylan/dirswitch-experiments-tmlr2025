import dataclasses as dc
from typing import Literal, Optional

import torch

import nebtools.data.graph as dgraph
from nebtools.ssgnns.utils import SSGNNTrainer, get_features

from .CCASSG.model import CCA_SSG
from .CCASSG.aug import random_aug


@dc.dataclass(frozen=True)
class Parameters:
    lr: float = 1e-3
    wd: float = 0.0
    lambd: float = 1e-3
    n_layers: int = 2
    drop_edge_ratio: float = 0.2
    drop_feature_ratio: float = 0.2
    hid_dim: int = 512
    out_dim: int = 512
    model_type: Literal[
        "gcn", "mlp", "graphsage_rossi", "graphsage_switch", "gcn_rossi", "gcn_switch"
    ] = "gcn_rossi"
    dir_seqs: Optional[tuple[str]] = None
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True

    # @staticmethod
    # def add_args(parser):
    #     # parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    #     # parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    #     # parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    #     parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of CCA-SSG.')
    #     parser.add_argument('--wd', type=float, default=0, help='Weight decay of CCA-SSG.')
    #
    #     parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')
    #     parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    #
    #     parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
    #     parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')
    #
    #     parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
    #     parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')
    #     parser.add_argument("--no_degree", action="store_true", help='Degree features not added.')
    #     parser.add_argument("--no_lcc", action="store_true", help='LCC features not added.')
    #     return parser
    #
    # @classmethod
    # def from_args(cls, args):
    #     params = cls(
    #         lr=args.lr,
    #         wd=args.wd,
    #         lambd=args.lambd,
    #         n_layers=args.n_layers,
    #         drop_edge_ratio=args.der,
    #         drop_feature_ratio=args.dfr,
    #         hid_dim=args.hid_dim,
    #         out_dim=args.out_dim,
    #         add_degree=not args.no_degree,
    #         add_lcc=not args.no_lcc,
    #     )
    #     return params


class CCASSGTrainer(SSGNNTrainer):
    def __init__(
        self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device
    ):
        self.graph = graph.to_dgl_graph()
        if params.model_type.startswith("gcn"):
            self.graph = self.graph.remove_self_loop().add_self_loop()
        else:
            self.graph = self.graph.remove_self_loop()
        self.feat = get_features(
            graph,
            add_degree=params.add_degree,
            add_lcc=params.add_lcc,
            standardize=params.standardize,
        )
        self.params = params
        self.num_nodes = graph.num_nodes
        in_dim = self.feat.shape[1]
        self.model = CCA_SSG(
            in_dim,
            params.hid_dim,
            params.out_dim,
            params.n_layers,
            directed=graph.directed,
            model_type=params.model_type,
            dir_seqs=params.dir_seqs,
        )
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, weight_decay=params.wd
        )
        self.device = device

    def step(self, step: int):
        self.model.train()
        self.optimizer.zero_grad()

        graph1, feat1 = random_aug(
            self.graph,
            self.feat,
            self.params.drop_feature_ratio,
            self.params.drop_edge_ratio,
        )
        graph2, feat2 = random_aug(
            self.graph,
            self.feat,
            self.params.drop_feature_ratio,
            self.params.drop_edge_ratio,
        )

        if self.params.model_type.startswith("gcn"):
            graph1 = graph1.add_self_loop()
            graph2 = graph2.add_self_loop()

        graph1 = graph1.to(self.device)
        graph2 = graph2.to(self.device)

        feat1 = feat1.to(self.device)
        feat2 = feat2.to(self.device)

        z1, z2 = self.model(graph1, feat1, graph2, feat2)

        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        c = c / self.num_nodes
        c1 = c1 / self.num_nodes
        c2 = c2 / self.num_nodes

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.eye(c.shape[0], device=self.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + self.params.lambd * (loss_dec1 + loss_dec2)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        self.model.eval()
        embeds = self.model.get_embedding(
            self.graph.to(self.device), self.feat.to(self.device)
        )
        return embeds

    @torch.no_grad()
    def get_embeddings_for_graph(self, graph: dgraph.SimpleGraph) -> torch.Tensor:
        dgl_graph = graph.to_dgl_graph()
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()
        feat = get_features(
            graph,
            add_degree=self.params.add_degree,
            add_lcc=self.params.add_lcc,
            standardize=self.params.standardize,
        )
        self.model.eval()
        embeds = self.model.get_embedding(
            dgl_graph.to(self.device), feat.to(self.device)
        )
        return embeds
