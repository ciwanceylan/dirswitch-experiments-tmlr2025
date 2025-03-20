"""
Adapted from the pytorch_geometric implementation of Node2Vec

Modified by Ciwan Ceylan, 2024-02
"""

from typing import List, Optional, Tuple, Union
import dataclasses as dc

import torch
from torch import Tensor
from torch.nn import Embedding, Parameter
from torch.utils.data import DataLoader

from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr

import reachnes.coeffs as rn_coeffs


@dc.dataclass(frozen=True)
class RNRWParams:
    num_nodes: int
    sub_embedding_dim: int
    use_forward_edges: bool
    use_reverse_edges: bool
    sampled_walk_length: int = 20
    walks_per_node: int = 1
    num_negative_samples: int = 1
    sparse: bool = True
    cpu_workers: int = 8
    batch_size: int = 128
    lr: float = 0.01


class ReachnesRW(torch.nn.Module):
    r"""The ReachnesRW model

    Args:
        edge_index (torch.Tensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        sampled_walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`True`)
    """

    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        reverse_edges: bool,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        sampled_walk_length: int = 20,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = True,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        dtype = torch.float32 if dtype is None else dtype

        if WITH_PYG_LIB:
            self.random_walk_fn = torch.ops.pyg.random_walk
        elif WITH_TORCH_CLUSTER:
            self.random_walk_fn = torch.ops.torch_cluster.random_walk
        else:
            raise ImportError(
                f"'{self.__class__.__name__}' "
                f"requires either the 'pyg-lib' or "
                f"'torch-cluster' package"
            )
        assert edge_index.shape[0] == 2
        self.reverse_edges = reverse_edges
        if reverse_edges:
            edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0)

        coeffs_obj = coeffs_obj.to(dtype=dtype)
        coefficients = coeffs_obj()
        assert coefficients.shape[0] == 1
        self.coefficients = coefficients[0]
        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        context_size = len(self.coefficients)
        self.EPS = 1e-15
        assert sampled_walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = sampled_walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples

        self.src_embeddings = Embedding(
            self.num_nodes, embedding_dim, sparse=sparse, dtype=dtype
        )
        self.dst_embeddings = Embedding(
            self.num_nodes, embedding_dim, sparse=sparse, dtype=dtype
        )
        self._dummy_param = Parameter(torch.empty(0))

        self.reset_parameters()

    def set_embeddings(self, src_embeddings=None, dst_embeddings=None):
        if src_embeddings is not None:
            self._set_embeddings(src_embeddings, is_src=True)

        if dst_embeddings is not None:
            self._set_embeddings(dst_embeddings, is_src=False)

    def _set_embeddings(self, embeddings, is_src: bool):
        if is_src and not self.reverse_edges:
            target_module = self.src_embeddings
        elif is_src and self.reverse_edges:
            target_module = self.dst_embeddings
        elif not is_src and not self.reverse_edges:
            target_module = self.dst_embeddings
        else:
            target_module = self.src_embeddings
        assert embeddings.shape == target_module.weight.shape
        dtype = target_module.weight.data.dtype
        device = target_module.weight.data.device
        target_module.weight.data = embeddings.clone().to(dtype=dtype, device=device)

    @property
    def current_device(self):
        return self._dummy_param.device

    # def coefficients(self) -> Tensor:
    #     if self.coeffs_obj.fixed_coeffs:
    #         return self._init_coeffs
    #     else:
    #         return self.coeffs_obj()[0]

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.src_embeddings.reset_parameters()
        self.dst_embeddings.reset_parameters()

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        src_embs = self.src_embeddings(batch)
        dst_embs = self.dst_embeddings(batch)
        return src_embs, dst_embs

    @torch.no_grad()
    def get_reverse_corrected_embeddings(self):
        """Returns the embeddings but corrected if 'reversed_edges' is used.
        That is, the method will always return embeddings in the format:
        (source embeddings, destination embeddings)."""
        batch = torch.arange(
            self.num_nodes, dtype=torch.long, device=self.current_device
        )
        src_embeddings, dst_embeddings = self(batch=batch)
        if self.reverse_edges:
            # Reverse the order since the random walk is performed in reverse.
            return dst_embeddings, src_embeddings
        else:
            return src_embeddings, dst_embeddings

    def loader(self, **kwargs) -> DataLoader:
        return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(
            self.rowptr, self.col, batch, self.walk_length, 1.0, 1.0
        )
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size + 1])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(
            self.num_nodes,
            (batch.size(0), self.walk_length),
            dtype=batch.dtype,
            device=batch.device,
        )
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j : j + self.context_size + 1])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.src_embeddings(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.dst_embeddings(rest.view(-1)).view(
            pos_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1)
        loss_logits = -torch.log(torch.sigmoid(out) + self.EPS)

        pos_loss = torch.einsum("ij,j->i", loss_logits, self.coefficients).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.src_embeddings(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.dst_embeddings(rest.view(-1)).view(
            neg_rw.size(0), -1, self.embedding_dim
        )

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.src_embeddings.weight.size(0)}, "
            f"{self.src_embeddings.weight.size(1)}+{self.dst_embeddings.weight.size(1)})"
        )


class RNRWTrainer:
    params: RNRWParams
    current_device: torch.device
    fwd_rnrw_model: ReachnesRW = None
    fwd_optimizer: torch.optim.Optimizer = None
    bwd_rnrw_model: ReachnesRW = None
    bwd_optimizer: torch.optim.Optimizer = None

    def __init__(
        self,
        edge_index: Tensor,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        params: RNRWParams,
        device: torch.device,
    ):
        self.current_device = device
        self.params = params

        if params.use_forward_edges:
            self.fwd_rnrw_model = ReachnesRW(
                edge_index=edge_index,
                embedding_dim=params.sub_embedding_dim,
                reverse_edges=False,
                coeffs_obj=coeffs_obj,
                sampled_walk_length=params.sampled_walk_length,
                walks_per_node=params.walks_per_node,
                num_negative_samples=params.num_negative_samples,
                num_nodes=params.num_nodes,
                sparse=params.sparse,
            ).to(device)
            if params.sparse:
                self.fwd_optimizer = torch.optim.SparseAdam(
                    list(self.fwd_rnrw_model.parameters()), lr=params.lr
                )
            else:
                self.fwd_optimizer = torch.optim.Adam(
                    list(self.fwd_rnrw_model.parameters()), lr=params.lr
                )

        if params.use_reverse_edges:
            self.bwd_rnrw_model = ReachnesRW(
                edge_index=edge_index,
                embedding_dim=params.sub_embedding_dim,
                reverse_edges=True,
                coeffs_obj=coeffs_obj,
                sampled_walk_length=params.sampled_walk_length,
                walks_per_node=params.walks_per_node,
                num_negative_samples=params.num_negative_samples,
                num_nodes=params.num_nodes,
                sparse=params.sparse,
            ).to(device)
            if params.sparse:
                self.bwd_optimizer = torch.optim.SparseAdam(
                    list(self.bwd_rnrw_model.parameters()), lr=params.lr
                )
            else:
                self.bwd_optimizer = torch.optim.Adam(
                    list(self.bwd_rnrw_model.parameters()), lr=params.lr
                )

    def get_fwd_embeddings(self):
        if self.fwd_rnrw_model is None:
            raise ValueError("Forward edges model not initialized.")
        return self.fwd_rnrw_model.get_reverse_corrected_embeddings()

    def get_bwd_embeddings(self):
        if self.bwd_rnrw_model is None:
            raise ValueError("Backward edges model not initialized.")
        return self.bwd_rnrw_model.get_reverse_corrected_embeddings()

    def get_embeddings(self):
        src_embeddings = []
        dst_embeddings = []
        if self.fwd_rnrw_model is not None:
            fwd_src_embeddings, fwd_dst_embeddings = (
                self.fwd_rnrw_model.get_reverse_corrected_embeddings()
            )
            src_embeddings.append(fwd_src_embeddings)
            dst_embeddings.append(fwd_dst_embeddings)
        if self.bwd_rnrw_model is not None:
            bwd_src_embeddings, bwd_dst_embeddings = (
                self.bwd_rnrw_model.get_reverse_corrected_embeddings()
            )
            src_embeddings.append(bwd_src_embeddings)
            dst_embeddings.append(bwd_dst_embeddings)
        src_embeddings = torch.cat(src_embeddings, dim=1)
        dst_embeddings = torch.cat(dst_embeddings, dim=1)
        return src_embeddings, dst_embeddings

    def get_concat_embeddings(self):
        src_embeddings, dst_embeddings = self.get_embeddings()
        return torch.cat((src_embeddings, dst_embeddings), dim=1)

    def set_embeddings(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        for_fwd_model: bool,
    ):
        if for_fwd_model:
            self.fwd_rnrw_model.set_embeddings(
                src_embeddings=src_embeddings, dst_embeddings=dst_embeddings
            )
        else:
            self.bwd_rnrw_model.set_embeddings(
                src_embeddings=src_embeddings, dst_embeddings=dst_embeddings
            )
        pass

    def set_split_embeddings(self, embeddings: torch.Tensor):
        src_embeddings, dst_embeddings = torch.chunk(embeddings, chunks=2, dim=1)
        if self.fwd_rnrw_model is not None and self.bwd_rnrw_model is not None:
            fwd_src_embeddings, bwd_src_embeddings = torch.chunk(
                src_embeddings, chunks=2, dim=1
            )
            fwd_dst_embeddings, bwd_dst_embeddings = torch.chunk(
                dst_embeddings, chunks=2, dim=1
            )
            self.set_embeddings(
                fwd_src_embeddings, fwd_dst_embeddings, for_fwd_model=True
            )
            self.set_embeddings(
                bwd_src_embeddings, bwd_dst_embeddings, for_fwd_model=False
            )
        elif self.fwd_rnrw_model is not None:
            self.set_embeddings(src_embeddings, dst_embeddings, for_fwd_model=True)
        elif self.bwd_rnrw_model is not None:
            self.set_embeddings(src_embeddings, dst_embeddings, for_fwd_model=False)
        else:
            raise ValueError("No embedding models available to set embeddings.")

    def run_training(self, num_epochs: int):
        fwd_loss = float("nan")
        bwd_loss = float("nan")
        for epoch in range(1, num_epochs + 1):
            if self.fwd_rnrw_model is not None:
                fwd_loss = self.train_epoch(
                    batch_size=self.params.batch_size,
                    num_cpu_workers=self.params.cpu_workers,
                    use_bwd_model=False,
                )
            if self.bwd_rnrw_model is not None:
                bwd_loss = self.train_epoch(
                    batch_size=self.params.batch_size,
                    num_cpu_workers=self.params.cpu_workers,
                    use_bwd_model=True,
                )
            print(
                f"Epoch {epoch:02d}, Fwd loss: {fwd_loss:.4f}, Bwd loss: {bwd_loss:.4f}"
            )
        return fwd_loss, bwd_loss

    def train_epoch(
        self, batch_size: int, num_cpu_workers: int, use_bwd_model: bool = False
    ):
        rnrw_model = self.bwd_rnrw_model if use_bwd_model else self.fwd_rnrw_model
        optimizer = self.bwd_optimizer if use_bwd_model else self.fwd_optimizer
        rnrw_model.train()
        loader = rnrw_model.loader(
            batch_size=batch_size, shuffle=True, num_workers=num_cpu_workers
        )
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = rnrw_model.loss(
                pos_rw.to(self.current_device), neg_rw.to(self.current_device)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss

    @torch.no_grad()
    def eval_loss(
        self, batch_size: int, num_cpu_workers: int, use_bwd_model: bool = False
    ):
        rnrw_model = self.bwd_rnrw_model if use_bwd_model else self.fwd_rnrw_model
        loader = rnrw_model.loader(
            batch_size=batch_size, shuffle=True, num_workers=num_cpu_workers
        )
        total_loss = 0
        for pos_rw, neg_rw in loader:
            loss = rnrw_model.loss(
                pos_rw.to(self.current_device), neg_rw.to(self.current_device)
            )
            total_loss += loss.item()
        return total_loss
