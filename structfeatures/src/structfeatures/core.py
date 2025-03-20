import os
import numba as nb
import numpy as np

USE_CACHE = bool(int(os.environ["STRUCTFEATURES_USE_NUMBA_CACHE"]))


@nb.jit(
    nb.types.Array(nb.types.int64, 1, "C", readonly=True)(
        nb.int64,
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
    cache=USE_CACHE,
)
def get_neigh(node: int, indices: np.ndarray, indptr: np.ndarray):
    return indices[indptr[node] : indptr[node + 1]]


@nb.jit(
    nb.types.Array(nb.types.int64, 1, "C", readonly=True)(
        nb.int64,
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
    cache=USE_CACHE,
)
def get_data(node: int, data: np.ndarray, indptr: np.ndarray):
    return data[indptr[node] : indptr[node + 1]]


@nb.jit(
    nb.types.Tuple((nb.int64[::1], nb.int64[::1], nb.int64[::1], nb.int64[::1]))(
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
    parallel=True,
    cache=USE_CACHE,
)
def _count_number_triangles_directed(
    out_indices: np.ndarray,
    out_indptr: np.ndarray,
    in_indices: np.ndarray,
    in_indptr: np.ndarray,
):
    """Count the number of out, in, cycle and middleman triangles based on the definitions in
    https://arxiv.org/pdf/physics/0612169.pdf.
    """
    num_nodes = len(out_indptr) - 1
    num_out_triangles = np.zeros((num_nodes,), dtype=np.int64)
    num_in_triangles = np.zeros((num_nodes,), dtype=np.int64)
    num_cycle_triangles = np.zeros((num_nodes,), dtype=np.int64)
    num_middle_triangles = np.zeros((num_nodes,), dtype=np.int64)
    for v in nb.prange(num_nodes):
        out_egonet = set(get_neigh(v, out_indices, out_indptr))
        in_egonet = set(get_neigh(v, in_indices, in_indptr))
        for out_neigh in out_egonet:
            for out_out_neigh in get_neigh(out_neigh, out_indices, out_indptr):
                if out_out_neigh == v:
                    continue
                if out_out_neigh in out_egonet:
                    num_out_triangles[v] += 1
                if out_out_neigh in in_egonet:
                    num_cycle_triangles[v] += 1
        for in_neigh in in_egonet:
            for out_in_neigh in get_neigh(in_neigh, out_indices, out_indptr):
                if out_in_neigh == v:
                    continue
                if out_in_neigh in out_egonet:
                    num_middle_triangles[v] += 1
                if out_in_neigh in in_egonet:
                    num_in_triangles[v] += 1
    return (
        num_out_triangles,
        num_in_triangles,
        num_cycle_triangles,
        num_middle_triangles,
    )


@nb.jit(
    nb.int64[::1](
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
    parallel=True,
    cache=USE_CACHE,
)
def _count_number_local_triangles(indices: np.ndarray, indptr: np.ndarray):
    num_nodes = len(indptr) - 1
    num_triangles = np.zeros((num_nodes,), dtype=np.int64)
    for v in nb.prange(num_nodes):
        egonet = set()
        egonet.update(get_neigh(v, indices, indptr))
        num_t = 0
        for neigh in egonet:
            for neigh_neigh in get_neigh(neigh, indices, indptr):
                if neigh_neigh in egonet:
                    num_t += 1
        num_triangles[v] = num_t
    return num_triangles


@nb.jit(
    nb.float64[::1](
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
    parallel=True,
    cache=USE_CACHE,
)
def two_hop_neighbours_homophily_coefficients(
    indices: np.ndarray, indptr: np.ndarray, node_labels: np.ndarray
):
    """Compute the homophiliy wrt nodes two steps away."""
    num_nodes = len(indptr) - 1
    assert len(node_labels) == num_nodes
    homophilies = np.zeros((num_nodes,), dtype=np.float64)
    for v in nb.prange(num_nodes):
        egonet = set(get_neigh(v, indices, indptr))
        self_label = node_labels[v]
        covered_2nd_order_neigh = set()
        homophily_coeff = 0.0
        total_2nd_neighbours = 0.0
        for neigh in egonet:
            neigh_egonet = get_neigh(neigh, indices, indptr)
            if len(neigh_egonet) == 1:
                homophily_coeff = np.nan
                continue
            else:
                for neigh_neigh in get_neigh(neigh, indices, indptr):
                    if (
                        neigh_neigh in covered_2nd_order_neigh
                        or neigh_neigh in egonet
                        or neigh_neigh == v
                    ):
                        continue
                    if node_labels[neigh_neigh] == self_label:
                        homophily_coeff += 1.0
                    total_2nd_neighbours += 1.0
                    covered_2nd_order_neigh.add(neigh_neigh)
        homophilies[v] = homophily_coeff / max(total_2nd_neighbours, 1e-6)

    return homophilies


@nb.jit(
    nb.float64[::1](
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
        nb.types.Array(nb.types.int64, 1, "C", readonly=True),
    ),
    nopython=True,
    nogil=True,
    parallel=True,
    cache=USE_CACHE,
)
def one_hop_neighbours_homophily_coefficients(
    indices: np.ndarray, indptr: np.ndarray, node_labels: np.ndarray
):
    """Compute the homophiliy wrt egonet nodes."""
    num_nodes = len(indptr) - 1
    assert len(node_labels) == num_nodes
    homophilies = np.zeros((num_nodes,), dtype=np.float64)
    for v in nb.prange(num_nodes):
        egonet = set(get_neigh(v, indices, indptr))
        self_label = node_labels[v]
        homophily_coeff = 0.0
        total_2nd_neighbours = 0.0
        for neigh in egonet:
            if node_labels[neigh] == self_label:
                homophily_coeff += 1.0
            total_2nd_neighbours += 1.0
        homophilies[v] = homophily_coeff / max(total_2nd_neighbours, 1e-6)

    return homophilies
