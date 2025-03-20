from typing import Union, Sequence, Tuple
import os
import errno
from itertools import groupby
import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse as tsp

AdjType = Union[np.ndarray, sp.spmatrix, sp.sparray, torch.Tensor, tsp.SparseTensor]
TorchAdjType = Union[torch.Tensor, tsp.SparseTensor]
MultiTorchAdjType = Union[torch.Tensor, Tuple[tsp.SparseTensor]]

numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}


class NotEnoughAvailableMemory(ValueError):
    pass


def sort_sparse_tensor_columns(mat_sp: tsp.SparseTensor) -> tsp.SparseTensor:
    num_rows, num_cols = mat_sp.sizes()
    value_dtype = mat_sp.dtype()
    rows = mat_sp.storage.row().to(torch.double)
    values = mat_sp.storage.value().to(torch.double)
    max_value = 2 * torch.abs(values).max() + 2
    _, new_order = torch.sort(-rows * max_value + values, descending=True)
    rows = rows[new_order].to(torch.long)
    values = values[new_order].to(value_dtype)
    new_cols = [
        torch.arange(len(list(group)), dtype=torch.long) for row, group in groupby(rows)
    ]
    new_cols = torch.cat(new_cols)
    out = tsp.SparseTensor(
        row=rows, col=new_cols, value=values, sparse_sizes=(num_rows, num_cols)
    )
    return out


def hacky_auto_batch_size(
    *,
    memory_available: int,
    num_nodes: int,
    num_edges: int,
    num_series: int,
    k_emb: int,
    bytes_per_element=4,
    cuda_overhead: float = 1.5,
):
    capacity = (memory_available - cuda_overhead) * (1024**3)

    needed_adj = 8 * (3 * (num_edges + num_nodes)) + bytes_per_element * (
        num_edges + num_nodes
    )
    needed_thresholds = 8 * num_nodes
    needed_embs = bytes_per_element * num_nodes * k_emb
    available_memory = capacity - needed_adj - needed_thresholds - needed_embs
    if available_memory <= 0:
        raise NotEnoughAvailableMemory(
            "Not enough memory to store adjacency matrix and embeddings"
        )
    # Needed for batch computation of expm. Requires some slack during computations.
    needed_dense_batch = 3.5 * num_nodes * (num_series + 3) * bytes_per_element
    batch_size = available_memory / needed_dense_batch
    num_needed_batches = np.ceil(num_nodes / batch_size)
    batch_size = int(np.ceil(num_nodes / num_needed_batches))
    return batch_size


def make_node_batches(
    num_nodes: int,
    batch_size: int,
    device: torch.device,
    node_indices: Sequence[int] = None,
):
    assert batch_size is not None
    assert batch_size > 0

    if node_indices is None:
        node_indices = torch.arange(num_nodes, dtype=torch.int64, device=device)
    elif isinstance(node_indices, torch.Tensor):
        node_indices = node_indices.to(dtype=torch.int64, device=device)
    else:
        node_indices = torch.tensor(node_indices, dtype=torch.int64, device=device)
    if node_indices.max() >= num_nodes:
        raise ValueError("Node index cannot be larger than number of nodes.")

    batches = [
        node_indices[b : b + batch_size].to(dtype=torch.int64, device=device).long()
        for b in range(0, len(node_indices), batch_size)
    ]
    return batches


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred
