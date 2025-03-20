import dataclasses as dc
import torch
import torch_geometric.sampler as pygsampler
import torch_geometric.utils as pygutils
import torch.nn.functional as tF
import torch_sparse as tsp

import nebtools.ssgnns.blade.BLADE.model as blade


@dc.dataclass(frozen=True)
class BladeSetup:
    edge_score: torch.Tensor
    alpha: float
    num_pos_per_node: torch.Tensor
    num_neg_per_positive: int = 1

    @classmethod
    def create_from_graph(cls, edge_index: torch.Tensor, num_nodes: int):
        out_degree = pygutils.degree(
            edge_index[0], num_nodes=num_nodes, dtype=torch.float
        )
        in_degree = pygutils.degree(
            edge_index[1], num_nodes=num_nodes, dtype=torch.float
        )
        in_degree_plus = torch.maximum(in_degree, torch.tensor(1))
        d_min = torch.min(in_degree)
        alpha_denominator = torch.sum(torch.log(in_degree_plus / d_min))
        alpha = 1 + (num_nodes / alpha_denominator)


class BLADELoss:
    def __init__(self, neg_per_pos: int = 1, use_pos_edge_score: bool = True):
        self.neg_per_pos = neg_per_pos
        self.num_nodes = None
        self.pos_edge_index = None
        self.pos_edge_score = None
        self.asym_edge_index = None
        self.asym_edge_scores = None
        self.use_pos_edge_score = use_pos_edge_score

    def set_positive_edge_index(self, edge_index: torch.Tensor, num_nodes: int):
        self.num_nodes = num_nodes
        self.pos_edge_index = edge_index
        escore = compute_edge_score(edge_index, num_nodes=num_nodes)
        self.pos_edge_score = escore / torch.sum(escore)
        self.asym_edge_index, self.asym_edge_scores = self._get_asym_edges_and_scores(
            edge_index=edge_index, num_nodes=num_nodes, edge_scores=self.pos_edge_score
        )

    # def to(self, device: torch.device):
    #     self.

    @staticmethod
    def _get_asym_edges_and_scores(
        edge_index: torch.Tensor, num_nodes: int, edge_scores: torch.Tensor
    ):
        adj_with_scores = tsp.SparseTensor.from_edge_index(
            edge_index=edge_index,
            edge_attr=edge_scores,
            sparse_sizes=(num_nodes, num_nodes),
        )
        adj_t = adj_with_scores.t()
        neg_adj_t = adj_t.set_value(-1 * adj_t.storage.value(), layout="coo")
        adj_diff = adj_with_scores + neg_adj_t
        asym_edge_mask = adj_diff.storage.value() > 0
        asym_edge_src = adj_diff.storage.row()[asym_edge_mask]
        asym_edge_dst = adj_diff.storage.col()[asym_edge_mask]
        asym_edge_index = torch.stack((asym_edge_src, asym_edge_dst), dim=0)
        asym_edge_scores = adj_diff.storage.value()[asym_edge_mask]
        return asym_edge_index, asym_edge_scores

    # @property
    # def pos_edge_index(self) -> torch.Tensor:
    #     if self._pos_edge_index is None:
    #         raise ValueError(f"BLADETrainer must be initialized with 'set_positive_edge_index' before training.")
    #     return self._pos_edge_index
    #
    # @property
    # def asym_edge_index(self) -> torch.Tensor:
    #     if self._asym_edge_index is None:
    #         raise ValueError(f"BLADETrainer must be initialized with 'set_positive_edge_index' before training.")
    #     return self._asym_edge_index
    #
    # @property
    # def asym_edge_scores(self) -> torch.Tensor:
    #     if self._asym_edge_scores is None:
    #         raise ValueError(f"BLADETrainer must be initialized with 'set_positive_edge_index' before training.")
    #     return self._asym_edge_scores
    #
    # @property
    # def pos_adj(self) -> torch.Tensor:
    #     if self._pos_adj is None:
    #         raise ValueError(f"BLADETrainer must be initialized with 'set_positive_edge_index' before training.")
    #     return self._pos_adj
    #
    # @property
    # def pos_edge_score(self) -> torch.Tensor:
    #     if self._pos_edge_score is None:
    #         raise ValueError(f"BLADETrainer must be initialized with 'set_positive_edge_index' before training.")
    #     return self._pos_edge_index
    #
    # @property
    # def num_nodes(self) -> int:
    #     if self._num_nodes is None:
    #         raise ValueError(f"BLADETrainer must be initialized with 'set_positive_edge_index' before training.")
    #     return self._num_nodes

    def posloss(self, emb_s: torch.Tensor, emb_t: torch.Tensor):
        emb_s_u = emb_s[self.pos_edge_index[0]]
        emb_t_v = emb_t[self.pos_edge_index[1]]
        loss = tF.logsigmoid(torch.einsum("ij,ij->i", emb_s_u, emb_t_v))
        if self.use_pos_edge_score:
            loss = torch.dot(self.pos_edge_score, loss)
        else:
            loss = torch.sum(loss)
        loss = -1 * loss
        return loss

    def sample_neg_edges(self):
        num_neg = self.neg_per_pos * self.pos_edge_index.shape[1]
        neg_edge_index = pygutils.negative_sampling(
            edge_index=self.pos_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=num_neg,
        )
        return neg_edge_index

    def negloss(
        self, emb_s: torch.Tensor, emb_t: torch.Tensor, neg_edge_index: torch.Tensor
    ):
        emb_s_u = emb_s[neg_edge_index[0]]
        emb_t_v = emb_t[neg_edge_index[1]]
        loss = tF.logsigmoid(1 - torch.einsum("ij,ij->i", emb_s_u, emb_t_v))
        loss = -1 * torch.mean(loss)
        return loss

    def asymposloss(self, emb_s: torch.Tensor, emb_t: torch.Tensor):
        emb_s_u = emb_s[self.asym_edge_index[0]]
        emb_t_v = emb_t[self.asym_edge_index[1]]
        loss = tF.logsigmoid(torch.einsum("ij,ij->i", emb_s_u, emb_t_v))
        if self.use_pos_edge_score:
            loss = torch.dot(self.asym_edge_scores, loss)
        else:
            loss = torch.sum(loss)
        loss = -1 * loss
        return loss

    def asymnegloss(self, emb_s: torch.Tensor, emb_t: torch.Tensor):
        emb_s_v = emb_s[self.asym_edge_index[1]]
        emb_t_u = emb_t[self.asym_edge_index[0]]
        loss = tF.logsigmoid(1 - torch.einsum("ij,ij->i", emb_s_v, emb_t_u))
        if self.use_pos_edge_score:
            loss = torch.dot(self.asym_edge_scores, loss)
        else:
            loss = torch.sum(loss)
        loss = -1 * loss
        return loss

    def loss(self, emb_s: torch.Tensor, emb_t: torch.Tensor):
        neg_edge_index = self.sample_neg_edges()
        posloss = self.posloss(emb_s=emb_s, emb_t=emb_t)
        negloss = self.negloss(emb_s=emb_s, emb_t=emb_t, neg_edge_index=neg_edge_index)
        asymposloss = self.asymposloss(emb_s=emb_s, emb_t=emb_t)
        asymnegloss = self.asymnegloss(emb_s=emb_s, emb_t=emb_t)
        return posloss + negloss + asymposloss + asymnegloss


def compute_edge_score(edge_index: torch.Tensor, num_nodes: int):
    out_degree = pygutils.degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    in_degree = pygutils.degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float)
    score = out_degree[edge_index[0]] * in_degree[edge_index[1]]
    return score
