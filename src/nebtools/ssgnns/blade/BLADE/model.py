from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as tF
import torch_sparse as tsp
import torch_geometric.nn as pygnn


class BLADENet(pygnn.MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__(aggr="sum")
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        lin_layers = [
            nn.Linear(in_features=in_dim if i == 0 else out_dim, out_features=out_dim)
            for i in range(num_layers)
        ]
        self.lin_layers = nn.ModuleList(lin_layers)

    def forward(self, x: torch.Tensor, adj: tsp.SparseTensor):
        h_s = x
        h_t = x

        for l, layer in enumerate(self.lin_layers):
            new_h_s = self.propagate(adj.t(), x=h_t)
            new_h_s = tF.relu(layer(new_h_s))

            new_h_t = self.propagate(adj, x=h_s)
            new_h_t = tF.relu(layer(new_h_t))

            h_s_norm = torch.maximum(
                torch.linalg.vector_norm(new_h_s, dim=0, keepdim=True),
                torch.tensor(1e-6),
            )
            new_h_s = new_h_s / h_s_norm
            h_t_norm = torch.maximum(
                torch.linalg.vector_norm(new_h_t, dim=0, keepdim=True),
                torch.tensor(1e-6),
            )
            new_h_t = new_h_t / h_t_norm

            h_s = new_h_s
            h_t = new_h_t
        return h_s, h_t

    @torch.no_grad()
    def embed(self, x: torch.Tensor, adj: tsp.SparseTensor):
        return self(x=x, adj=adj)

    def message_and_aggregate(self, adj: tsp.SparseTensor, x: torch.Tensor):
        return tsp.matmul(adj, x, reduce=self.aggr)

    def initialize_features(self, method: Literal["normal"], num_nodes: int):
        match method:
            case "normal":
                x = torch.randn(num_nodes, self.in_dim)
            case _:
                raise ValueError(f"Invalid init method '{method}'.")
        return x
