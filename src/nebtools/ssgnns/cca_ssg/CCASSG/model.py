from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm="both"))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm="both"))
            self.convs.append(GraphConv(hid_dim, out_dim, norm="both"))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x


class DirGCN_rossi(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, alpha: float = 0.5):
        super().__init__()

        self.alpha = alpha
        self.n_layers = n_layers
        self.fwd_convs = nn.ModuleList()
        self.bwd_convs = nn.ModuleList()

        self.fwd_convs.append(GraphConv(in_dim, hid_dim, norm="both"))
        self.bwd_convs.append(GraphConv(in_dim, hid_dim, norm="both"))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.fwd_convs.append(GraphConv(hid_dim, hid_dim, norm="both"))
                self.bwd_convs.append(GraphConv(hid_dim, hid_dim, norm="both"))
            self.fwd_convs.append(GraphConv(hid_dim, out_dim, norm="both"))
            self.bwd_convs.append(GraphConv(hid_dim, out_dim, norm="both"))

    def forward(self, graph, x):
        rev_graph = graph.reverse()
        for i in range(self.n_layers - 1):
            x = F.relu(
                self.alpha * self.fwd_convs[i](graph, x)
                + (1 - self.alpha) * self.bwd_convs[i](rev_graph, x)
            )
        x = self.alpha * self.fwd_convs[-1](graph, x) + (
            1 - self.alpha
        ) * self.bwd_convs[-1](rev_graph, x)

        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.fwd_convs = nn.ModuleList()

        self.fc_layers = nn.ModuleList()

        self.fwd_convs.append(
            GraphConv(
                in_dim,
                hid_dim,
                norm="left",
                weight=False,
                bias=False,
                allow_zero_in_degree=True,
            )
        )
        self.fc_layers.append(nn.Linear(2 * in_dim, hid_dim))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.fwd_convs.append(
                    GraphConv(
                        hid_dim,
                        hid_dim,
                        norm="left",
                        weight=False,
                        bias=False,
                        allow_zero_in_degree=True,
                    )
                )
                self.fc_layers.append(nn.Linear(2 * hid_dim, hid_dim))
            self.fwd_convs.append(
                GraphConv(
                    hid_dim,
                    out_dim,
                    norm="left",
                    weight=False,
                    bias=False,
                    allow_zero_in_degree=True,
                )
            )
            self.fc_layers.append(nn.Linear(2 * hid_dim, out_dim))

    def forward(self, graph, x):
        for i in range(self.n_layers - 1):
            intermediate = torch.concat((x, self.fwd_convs[i](graph, x)), dim=1)
            x = F.relu(self.fc_layers[i](intermediate))
            # x = F.relu(self.alpha * self.fwd_convs[i](graph, x) + (1 - self.alpha) * self.bwd_convs[i](rev_graph, x))
        intermediate = torch.concat((x, self.fwd_convs[-1](graph, x)), dim=1)
        x = self.fc_layers[-1](intermediate)

        return x


class DirGraphSAGE_Rossi(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, alpha: float = 0.5):
        super().__init__()

        self.alpha = alpha
        self.n_layers = n_layers

        self.fwd_convs = nn.ModuleList()
        self.bwd_convs = nn.ModuleList()

        self.fc_layers = nn.ModuleList()

        self.fwd_convs.append(
            GraphConv(
                in_dim,
                in_dim,
                norm="left",
                weight=False,
                bias=False,
                allow_zero_in_degree=True,
            )
        )
        self.bwd_convs.append(
            GraphConv(
                in_dim,
                in_dim,
                norm="left",
                weight=False,
                bias=False,
                allow_zero_in_degree=True,
            )
        )
        self.fc_layers.append(nn.Linear(3 * in_dim, hid_dim))
        if n_layers > 1:
            for i in range(n_layers - 2):
                self.fwd_convs.append(
                    GraphConv(
                        hid_dim,
                        hid_dim,
                        norm="left",
                        weight=False,
                        bias=False,
                        allow_zero_in_degree=True,
                    )
                )
                self.bwd_convs.append(
                    GraphConv(
                        hid_dim,
                        hid_dim,
                        norm="left",
                        weight=False,
                        bias=False,
                        allow_zero_in_degree=True,
                    )
                )
                self.fc_layers.append(nn.Linear(3 * hid_dim, hid_dim))
            self.fwd_convs.append(
                GraphConv(
                    hid_dim,
                    hid_dim,
                    norm="left",
                    weight=False,
                    bias=False,
                    allow_zero_in_degree=True,
                )
            )
            self.bwd_convs.append(
                GraphConv(
                    hid_dim,
                    hid_dim,
                    norm="left",
                    weight=False,
                    bias=False,
                    allow_zero_in_degree=True,
                )
            )
            self.fc_layers.append(nn.Linear(3 * hid_dim, out_dim))

    def forward(self, graph, x):
        rev_graph = graph.reverse()
        for i in range(self.n_layers - 1):
            intermediate = torch.concat(
                (x, self.fwd_convs[i](graph, x), self.bwd_convs[i](rev_graph, x)), dim=1
            )
            x = F.relu(self.fc_layers[i](intermediate))
        intermediate = torch.concat(
            (x, self.fwd_convs[-1](graph, x), self.bwd_convs[-1](rev_graph, x)), dim=1
        )
        x = self.fc_layers[-1](intermediate)

        return x


class GraphSAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim, use_activation: bool):
        super().__init__()
        self.convs = GraphConv(
            in_dim,
            in_dim,
            norm="left",
            weight=False,
            bias=False,
            allow_zero_in_degree=True,
        )
        self.linear = nn.Linear(2 * in_dim, out_dim)
        self.use_activation = use_activation

    def forward(self, graph, x):
        x = self.linear(torch.cat((x, self.convs(graph, x)), dim=1))
        if self.use_activation:
            x = F.relu(x)
        return x


def pad_dir_seq(dir_seq, num_layers):
    if len(dir_seq) > num_layers:
        raise ValueError(
            f"Length of dir_seq '{dir_seq}' exceeds number of layers, '{num_layers}'."
        )
    num_pad = num_layers - len(dir_seq)
    dir_seq = dir_seq + dir_seq[-1] * num_pad
    return dir_seq


class SwitchModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        n_layers,
        dir_seq,
        encoder: Literal["gcn", "graphsage"],
    ):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.dir_seq = pad_dir_seq(dir_seq, n_layers)

        match encoder:
            case "graphsage":
                self.convs.append(
                    GraphSAGEConv(
                        in_dim=in_dim, out_dim=hid_dim, use_activation=n_layers > 1
                    )
                )

                if n_layers > 1:
                    for i in range(n_layers - 2):
                        self.convs.append(
                            GraphSAGEConv(
                                in_dim=hid_dim, out_dim=hid_dim, use_activation=True
                            )
                        )

                    self.convs.append(
                        GraphSAGEConv(
                            in_dim=hid_dim, out_dim=out_dim, use_activation=False
                        )
                    )
            case "gcn":
                activation = F.relu if n_layers > 1 else None
                self.convs.append(
                    GraphConv(
                        in_feats=in_dim,
                        out_feats=hid_dim,
                        norm="both",
                        activation=activation,
                    )
                )
                if n_layers > 1:
                    for i in range(n_layers - 2):
                        self.convs.append(
                            GraphConv(
                                in_feats=hid_dim,
                                out_feats=hid_dim,
                                norm="both",
                                activation=F.relu,
                            )
                        )

                    self.convs.append(
                        GraphConv(
                            in_feats=hid_dim,
                            out_feats=out_dim,
                            norm="both",
                            activation=None,
                        )
                    )
            case _:
                raise ValueError(f"Unknown encoder '{encoder}'")

    def forward(self, graph, graph_undir, graph_reverse, x):
        for i, dir_spec in enumerate(self.dir_seq):
            match dir_spec:
                case "O":
                    x = self.convs[i](graph, x)
                case "I":
                    x = self.convs[i](graph_reverse, x)
                case "U":
                    x = self.convs[i](graph_undir, x)
                case _:
                    raise ValueError(f'Unknown dir spec "{dir_spec}"')
        return x


class MultiSwitchModel(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        n_layers,
        dir_seqs,
        encoder: Literal["gcn", "graphsage"],
    ):
        super().__init__()
        num_seq_heads = len(dir_seqs)
        self.gs_modules = torch.nn.ModuleList(
            [
                SwitchModel(
                    dir_seq=dir_seq,
                    in_dim=in_dim,
                    hid_dim=max(hid_dim // num_seq_heads, 2),
                    out_dim=max(out_dim // num_seq_heads, 1),
                    n_layers=n_layers,
                    encoder=encoder,
                )
                for dir_seq in dir_seqs
            ]
        )

    def forward(self, graph, inputs):
        embeddings = list()
        graph_undir = dgl.add_reverse_edges(g=graph)
        graph_rev = graph.reverse()

        for gs_model in self.gs_modules:
            h = gs_model(graph, graph_undir, graph_rev, inputs)
            embeddings.append(h)
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings


class CCA_SSG(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        n_layers,
        directed: bool,
        model_type: Literal["gcn", "mlp", "graphsage_rossi", "graphsage_switch"],
        dir_seqs,
    ):
        super().__init__()
        match model_type:
            case "mlp":
                self.backbone = MLP(in_dim, hid_dim, out_dim)
            case "gcn_rossi":
                if directed:
                    self.backbone = DirGCN_rossi(in_dim, hid_dim, out_dim, n_layers)
                else:
                    self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
            case "gcn_switch":
                if directed:
                    self.backbone = MultiSwitchModel(
                        in_dim,
                        hid_dim,
                        out_dim,
                        n_layers,
                        dir_seqs=dir_seqs,
                        encoder="gcn",
                    )
                else:
                    self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
            case "graphsage_rossi":
                if directed:
                    self.backbone = DirGraphSAGE_Rossi(
                        in_dim, hid_dim, out_dim, n_layers
                    )
                else:
                    self.backbone = GraphSAGE(in_dim, hid_dim, out_dim, n_layers)
            case "graphsage_switch":
                if directed:
                    self.backbone = MultiSwitchModel(
                        in_dim,
                        hid_dim,
                        out_dim,
                        n_layers,
                        dir_seqs=dir_seqs,
                        encoder="graphsage",
                    )
                else:
                    self.backbone = GraphSAGE(in_dim, hid_dim, out_dim, n_layers)
            case _:
                raise ValueError(f"Unknown model type '{model_type}'")

    def get_embedding(self, graph, feat):
        out = self.backbone(graph, feat)
        return out.detach()

    def forward(self, graph1, feat1, graph2, feat2):
        h1 = self.backbone(graph1, feat1)
        h2 = self.backbone(graph2, feat2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2
