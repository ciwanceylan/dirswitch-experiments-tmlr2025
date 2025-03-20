from typing import Sequence, Literal, Tuple

import dgl
import torch
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from nebtools.ssgnns.graphmae2.GraphMAE2.utils import create_activation, create_norm
from nebtools.ssgnns.graphmae2.GraphMAE2.models.dirgat import OneDirGATConv


class GATMultiSwitchModel(nn.Module):
    def __init__(
        self,
        dir_seqs: Tuple[Sequence[Literal["O", "I", "U"]]],
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        nhead,
        nhead_out,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        norm,
        concat_out=False,
        encoding=False,
    ):
        assert isinstance(dir_seqs, tuple)
        super(GATMultiSwitchModel, self).__init__()
        num_seq_heads = len(dir_seqs)

        self.gs_modules = torch.nn.ModuleList(
            [
                GATSwitchModel(
                    dir_seq=dir_seq,
                    in_dim=in_dim,
                    num_hidden=max(num_hidden // num_seq_heads, 2),
                    out_dim=max(out_dim // (num_seq_heads * nhead_out), 1),
                    nhead=nhead,
                    nhead_out=nhead_out,
                    num_layers=num_layers,
                    activation=activation,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    concat_out=concat_out,
                    norm=norm,
                    encoding=encoding,
                )
                for dir_seq in dir_seqs
            ]
        )

    def forward(self, g, inputs):
        embeddings = list()
        g_undir = dgl.add_reverse_edges(g=g)
        g_rev = g.reverse()

        for gs_model in self.gs_modules:
            h = gs_model(g, g_undir, g_rev, inputs)
            embeddings.append(h)
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings


def pad_dir_seq(dir_seq, num_layers):
    if len(dir_seq) > num_layers:
        raise ValueError(
            f"Length of dir_seq '{dir_seq}' exceeds number of layers, '{num_layers}'."
        )
    num_pad = num_layers - len(dir_seq)
    dir_seq = dir_seq + dir_seq[-1] * num_pad
    return dir_seq


class GATSwitchModel(nn.Module):
    def __init__(
        self,
        dir_seq: Sequence[Literal["O", "I", "U"]],
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        nhead,
        nhead_out,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        norm,
        concat_out=False,
        encoding=False,
    ):
        super(GATSwitchModel, self).__init__()
        self.dir_seq = pad_dir_seq(dir_seq, num_layers)
        self.out_dim = out_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.activation_fn = create_activation(activation)

        self.last_activation = create_activation(activation) if encoding else None
        last_norm = norm if encoding else None
        last_residual = encoding and residual

        hidden_in = in_dim
        hidden_out = out_dim

        if num_layers == 1:
            self.gat_layers.append(
                OneDirGATConv(
                    hidden_in,
                    hidden_out,
                    num_heads=nhead_out,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=last_residual,
                    norm=last_norm,
                    concat_out=concat_out,
                )
            )
        else:
            # input projection (no residual)
            self.gat_layers.append(
                OneDirGATConv(
                    in_feats=hidden_in,
                    out_feats=num_hidden,
                    num_heads=nhead,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    norm=norm,
                    concat_out=concat_out,
                )
            )
            # hidden layers

            for l in range(1, num_layers - 1):
                self.gat_layers.append(
                    OneDirGATConv(
                        in_feats=num_hidden * nhead,
                        out_feats=num_hidden,
                        num_heads=nhead,
                        feat_drop=feat_drop,
                        attn_drop=attn_drop,
                        negative_slope=negative_slope,
                        residual=residual,
                        norm=norm,
                        concat_out=concat_out,
                    )
                )

            # output projection
            self.gat_layers.append(
                OneDirGATConv(
                    in_feats=num_hidden * nhead,
                    out_feats=num_hidden,
                    num_heads=nhead_out,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=last_residual,
                    norm=last_norm,
                    concat_out=concat_out,
                )
            )
        self.head = nn.Identity()

    def forward(self, g, g_undir, g_rev, inputs):
        h = inputs

        for l, dir_spec in enumerate(self.dir_seq):
            if dir_spec == "O":
                h = self.gat_layers[l](g, h)
            elif dir_spec == "I":
                h = self.gat_layers[l](g_rev, h)
            elif dir_spec == "U":
                h = self.gat_layers[l](g_undir, h)
            else:
                raise ValueError(f"Unknown edge direction specifier '{dir_spec}'.")
            if l < len(self.dir_seq) - 1:
                h = self.activation_fn(h)
            elif self.last_activation is not None:
                h = self.last_activation(h)

        if self.head is not None:
            return self.head(h)
        else:
            return h

    # def reset_classifier(self, num_classes):
    #     self.num_classes = num_classes
    #     self.is_pretraining = False
    #     self.head = nn.Linear(self.out_dim, num_classes)
