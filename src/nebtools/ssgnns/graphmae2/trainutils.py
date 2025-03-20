import dataclasses as dc
from typing import Sequence, Literal, Tuple, Optional

import numpy as np
import torch

import nebtools.data.graph as dgraph
from nebtools.ssgnns.utils import SSGNNTrainer, get_features

from .GraphMAE2.models.edcoder import PreModel

ENCODERS = Literal[
    "graphsage_switch", "gat_switch", "gat_rossi", "graphsage_rossi", "mlp", "linear"
]


@dc.dataclass(frozen=True)
class Parameters:
    num_epochs: int = 1000
    lr: float = 0.001
    wd: float = 1e-4
    num_heads: int = 4
    num_hidden: int = 1024
    num_layers: int = 2
    mask_rate: float = 0.5
    replace_rate: float = 0.05
    alpha_l: int = 3
    lam: float = 1.0
    use_scheduler: bool = True
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True
    encoder: ENCODERS = "gat_switch"
    decoder: ENCODERS = "gat_switch"
    dir_seqs: Optional[Tuple[Sequence[Literal["O", "I", "U"]]]] = None

    # @staticmethod
    # def add_args(parser):
    #     parser.add_argument('--epochs', type=int, default=1000, help='Training epochs.')
    #     parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of GraphMAE.')
    #     parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay of GraphMAE.')
    #
    #     parser.add_argument('--num_heads', type=int, default=4, help='Number of GAT output heads')
    #     parser.add_argument('--num_hidden', type=int, default=512, help='Number hidden dimensions.')
    #     parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    #
    #     parser.add_argument('--mask_rate', type=float, default=0.5, help='Feature masking rate')
    #     parser.add_argument('--replace_rate', type=float, default=0.05, help='Replacement ratio.')
    #     parser.add_argument('--encoder', type=str, default='gat', help='Which encoder model to use.')
    #     parser.add_argument('--decoder', type=str, default='gat', help='Which decoder model to use.')
    #
    #     parser.add_argument("--alpha_l", type=int, default=3, help='Loss function scale factor (gamma).')
    #     parser.add_argument("--no_scheduler", action="store_true", help='Disable the learning rate scheduler.')
    #     parser.add_argument("--no_degree", action="store_true", help='Degree features not added.')
    #     parser.add_argument("--no_lcc", action="store_true", help='LCC features not added.')
    #     return parser
    #
    # @classmethod
    # def from_args(cls, args):
    #     use_scheduler = not args.no_scheduler
    #     params = cls(
    #         lr=args.lr,
    #         wd=args.wd,
    #         num_heads=args.num_heads,
    #         num_hidden=args.num_hidden,
    #         num_layers=args.num_layers,
    #         mask_rate=args.mask_rate,
    #         replace_rate=args.replace_rate,
    #         alpha_l=args.alpha_l,
    #         use_scheduler=use_scheduler,
    #         add_degree=not args.no_degree,
    #         add_lcc=not args.no_lcc,
    #     )
    #     return params


def build_model(
    directed: bool,
    num_in_features: int,
    params: Parameters,
    dir_seqs: Tuple[Sequence[Literal["O", "I", "U"]]],
):
    if params.encoder.startswith("graphsage"):
        num_heads = 1
    else:
        num_heads = params.num_heads

    if params.encoder.endswith("switch"):
        num_hidden = (num_heads * len(dir_seqs)) * (
            params.num_hidden // (num_heads * len(dir_seqs))
        )
    else:
        num_hidden = params.num_hidden

    num_layers = params.num_layers
    mask_rate = params.mask_rate
    replace_rate = params.replace_rate
    alpha_l = params.alpha_l
    lam = params.lam
    encoder_type = params.encoder
    decoder_type = params.decoder

    num_out_heads = 1
    norm = None
    residual = False
    attn_drop = 0.1
    in_drop = 0.2
    negative_slope = 0.2
    drop_edge_rate = 0.0
    activation = "prelu"
    loss_fn = "sce"
    remask_method = "fixed"
    mask_method = "random"
    remask_rate = 0.5
    momentum = 0.996
    num_remasking = 3
    num_dec_layers = 1
    delayed_ema_epoch = 0
    zero_init = False

    model = PreModel(
        directed=directed,
        dir_seqs=dir_seqs,
        in_dim=num_in_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        num_dec_layers=num_dec_layers,
        num_remasking=num_remasking,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        remask_rate=remask_rate,
        mask_method=mask_method,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        alpha_l=alpha_l,
        lam=lam,
        delayed_ema_epoch=delayed_ema_epoch,
        replace_rate=replace_rate,
        remask_method=remask_method,
        momentum=momentum,
        zero_init=zero_init,
    )
    return model


class GraphMAE2Trainer(SSGNNTrainer):
    def __init__(
        self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device
    ):
        self.graph = graph.to_dgl_graph()
        if params.encoder.startswith("graphsage"):
            self.graph = self.graph.remove_self_loop()
        else:
            self.graph = self.graph.remove_self_loop().add_self_loop()
        self.feat = get_features(
            graph,
            add_degree=params.add_degree,
            add_lcc=params.add_lcc,
            standardize=params.standardize,
        )
        self.graph = self.graph.to(device)
        self.feat = self.feat.to(device)
        self.target_nodes = torch.arange(
            self.feat.shape[0], device=device, dtype=torch.long
        )
        self.params = params
        self.num_nodes = graph.num_nodes
        dir_seqs = None if not graph.directed else params.dir_seqs
        self.model = build_model(
            directed=graph.directed,
            num_in_features=self.feat.shape[1],
            params=params,
            dir_seqs=dir_seqs,
        )
        self.model.to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, weight_decay=params.wd
        )

        if params.use_scheduler and params.num_epochs > 0:
            self.scheduler = (
                lambda epoch: (1 + np.cos((epoch) * np.pi / params.num_epochs)) * 0.5
            )
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=self.scheduler
            )
        else:
            self.scheduler = None

        self.device = device

    def step(self, step: int):
        self.model.train()

        loss = self.model(self.graph, self.feat, targets=self.target_nodes)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def get_embeddings(self) -> torch.Tensor:
        self.model.eval()
        embeds = self.model.embed(self.graph.to(self.device), self.feat.to(self.device))
        return embeds

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
        embeds = self.model.embed(dgl_graph.to(self.device), feat.to(self.device))
        return embeds
