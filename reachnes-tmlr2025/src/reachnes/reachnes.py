from typing import Optional, Sequence, Tuple, Union, Literal
import dataclasses as dc
import torch
import torch.nn as nn

import reachnes.reduction as rn_reduc
import reachnes.adj_utils as rn_adjutils
import reachnes.utils as rn_utils
import reachnes.coeffs as rn_coeffs
import reachnes.ew_filtering as rn_filtering
from reachnes.reachability import ReachabilityModel, ReachabilityTimesXModel


@dc.dataclass(frozen=True)
class RNParams:
    num_nodes: int
    nnz: int
    normalization_seq: Tuple[Sequence[rn_adjutils.AdjOrientation]]
    dtype: torch.dtype
    emb_dim: int
    batch_size: Union[int, Literal["auto"]] = "auto"
    memory_available: int = 8

    def __post_init__(self):
        assert isinstance(self.normalization_seq, tuple) or isinstance(
            self.normalization_seq, list
        )
        for n_seq in self.normalization_seq:
            _ = rn_adjutils.AdjSeq(n_seq)  # Test that sequence is valid

    def get_reduction_emb_dim(self, num_series: int, num_orientations: int):
        factor = num_series * num_orientations
        reduction_emb_dim = max(self.emb_dim // factor, 1)
        return reduction_emb_dim


class Reachnes(nn.Module):
    def __init__(
        self,
        params: RNParams,
        reduction_model: rn_reduc.ReductionModel,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        ew_filter: Optional[rn_filtering.FilterModel] = None,
    ):
        super().__init__()

        self.adj_seq = tuple(
            rn_adjutils.AdjSeq(n_seq) for n_seq in params.normalization_seq
        )
        self.reachability_model = ReachabilityModel(
            coeffs_obj=coeffs_obj, ew_filter=ew_filter
        )
        self.reduction_model = reduction_model
        self.reduction_model.init(
            num_series=coeffs_obj.num_series,
            emb_dim=params.get_reduction_emb_dim(
                num_series=coeffs_obj.num_series,
                num_orientations=len(params.normalization_seq),
            ),
            num_nodes=params.num_nodes,
        )

        bytes_per_element = torch.ones(1, dtype=params.dtype).element_size()
        if params.batch_size == "auto":
            self.batch_size = rn_utils.hacky_auto_batch_size(
                memory_available=params.memory_available,
                num_nodes=params.num_nodes,
                num_edges=params.nnz,
                num_series=coeffs_obj.num_series,
                k_emb=self.reduction_model.memory_factor(num_nodes=params.num_nodes),
                bytes_per_element=bytes_per_element,
                cuda_overhead=1.5,
            )
        else:
            self.batch_size = params.batch_size

    @property
    def coeffs_obj(self):
        return self.reachability_model.coeffs_obj

    def forward(
        self,
        adj_obj: rn_adjutils.TorchAdj,
        melt_embeddings: bool,
        node_indices: Sequence[int] = None,
    ):
        batches = rn_utils.make_node_batches(
            num_nodes=adj_obj.num_nodes,
            batch_size=self.batch_size,
            device=adj_obj.device,
            node_indices=node_indices,
        )
        embeddings = []
        embedding_node_order = None

        for adj_seq in self.adj_seq:
            pre_embs = rn_reduc.GatheredPreEmbeddings()
            for batch_indices in batches:
                reachability = self.reachability_model(
                    adj_obj, adj_seq=adj_seq, batch_indices=batch_indices
                )
                pre_embedding_batch = self.reduction_model.reachability2pre_embeddings(
                    reachability_batch=reachability, batch_indices=batch_indices
                )
                pre_embs.store_pre_embeddings(pre_embedding_batch)

            pre_embeddings = self.reduction_model.reduce_gathered_pre_embeddings(
                pre_embs
            )
            emb_series = self.reduction_model(pre_embeddings)
            embeddings.append(emb_series)
            embedding_node_order = pre_embeddings.batch_indices

        embeddings = torch.stack(embeddings, dim=0)
        if melt_embeddings:
            if self.reduction_model.is_proximal and getattr(
                self.reduction_model, "include_v", True
            ):
                embeddings = melt_proximal_embeddings(
                    embeddings, orientations=self.adj_seq
                )
            else:
                embeddings = melt_embedding_series(embeddings)

        return embeddings, embedding_node_order


class ReachnesDDP(Reachnes):
    def __init__(
        self,
        world_size: int,
        params: RNParams,
        reduction_model: rn_reduc.ReductionModel,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        ew_filter: Optional[rn_filtering.BetaincFilter] = None,
    ):
        super().__init__(
            params=params,
            reduction_model=reduction_model,
            coeffs_obj=coeffs_obj,
            ew_filter=ew_filter,
        )
        self.world_size = world_size

    def forward(
        self,
        adj_obj: rn_adjutils.TorchAdj,
        melt_embeddings: bool,
        node_indices: Sequence[int] = None,
    ):
        batches = rn_utils.make_node_batches(
            num_nodes=adj_obj.num_nodes,
            batch_size=self.batch_size,
            device=adj_obj.device,
            node_indices=node_indices,
        )

        embeddings = []
        embedding_node_order = None

        for adj_seq in self.adj_seq:
            gathered_pre_embs = rn_reduc.GatheredPreEmbeddings()
            for batch_indices in batches:
                reachability = self.reachability_model(
                    adj_obj, adj_seq=adj_seq, batch_indices=batch_indices
                )
                pre_embedding_batch = self.reduction_model.reachability2pre_embeddings(
                    reachability_batch=reachability, batch_indices=batch_indices
                )
                gathered_pre_embs.store_pre_embeddings(pre_embedding_batch)

            if self.reduction_model.requires_global_pre_emb_gather or bool(
                self.num_convs
            ):
                local_pre_embeddings = (
                    self.reduction_model.reduce_gathered_pre_embeddings(
                        gathered_pre_embs
                    )
                )
                # Merge local and global pre-embeddings
                gathered_pre_embs = (
                    rn_reduc.GatheredPreEmbeddings.tdistr_global_gather_pre_embs(
                        local_pre_embeddings, world_size=self.world_size
                    )
                )
            pre_embeddings = self.reduction_model.reduce_gathered_pre_embeddings(
                gathered_pre_embs
            )
            if embedding_node_order is None:
                embedding_node_order = pre_embeddings.batch_indices
            else:
                # TODO this is a sanity check that DDP returns in a consistent order. Can be removed once verified
                assert torch.allclose(
                    embedding_node_order, pre_embeddings.batch_indices
                )

            # Transform pre-embeddings to embeddings
            emb_series = self.reduction_model(pre_embeddings)
            embeddings.append(emb_series)

        embeddings = torch.stack(embeddings, dim=0)
        if melt_embeddings:
            if self.reduction_model.is_proximal and getattr(
                self.reduction_model, "include_v", True
            ):
                embeddings = melt_proximal_embeddings(
                    embeddings, orientations=self.adj_seq
                )
            else:
                embeddings = melt_embedding_series(embeddings)

        return embeddings, embedding_node_order


class ReachnesNodeAttributes(nn.Module):
    def __init__(
        self,
        params: RNParams,
        coeffs_obj: rn_coeffs.RWLCoefficientsModel,
        ew_filter: Optional[rn_filtering.FilterModel] = None,
    ):
        super().__init__()

        self.adj_seq = tuple(
            rn_adjutils.AdjSeq(n_seq) for n_seq in params.normalization_seq
        )
        self.reachability_model = ReachabilityTimesXModel(
            coeffs_obj=coeffs_obj, ew_filter=ew_filter
        )
        self.compress_dim = params.get_reduction_emb_dim(
            num_series=coeffs_obj.num_series,
            num_orientations=len(params.normalization_seq),
        )

    @property
    def coeffs_obj(self):
        return self.reachability_model.coeffs_obj

    def forward(
        self,
        adj_obj: rn_adjutils.TorchAdj,
        x: torch.Tensor,
        *,
        melt_embeddings: bool,
        no_compression: bool = False,
    ):
        embeddings = []
        use_compression = not no_compression

        for adj_seq in self.adj_seq:
            embedding_series = self.reachability_model(adj_obj, adj_seq=adj_seq, x=x)

            if use_compression:
                embedding_series, _ = rn_reduc.pca_compress_tensor(
                    x=embedding_series, num_dims=self.compress_dim
                )

            embeddings.append(embedding_series)
        embeddings = torch.stack(embeddings, dim=0)
        if melt_embeddings:
            embeddings = melt_embedding_series(embeddings)
        return embeddings


def melt_embedding_series(embeddings_series_per_orientation):
    num_orientations, num_series, num_nodes, emb_dim = (
        embeddings_series_per_orientation.shape
    )
    melted_embeddings = embeddings_series_per_orientation.permute(2, 0, 1, 3).reshape(
        num_nodes, num_orientations * num_series * emb_dim
    )
    return melted_embeddings


def melt_proximal_embeddings(
    embeddings_series_per_orientation: torch.Tensor,
    orientations: Tuple[rn_adjutils.AdjSeq],
):
    src_embs_series, dst_embs_series = rn_reduc.proximal_embedding_series2decomposition(
        embeddings_series_per_orientation
    )

    melted_src_embeddings = melt_embedding_series(src_embs_series)
    melted_dst_embeddings = melt_embedding_series(dst_embs_series)

    if orientations[0] in {"I", "B", "T"}:
        melted_embeddings = torch.cat(
            (melted_dst_embeddings, melted_src_embeddings), dim=1
        )
    else:
        melted_embeddings = torch.cat(
            (melted_src_embeddings, melted_dst_embeddings), dim=1
        )
    return melted_embeddings
