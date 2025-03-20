from typing import Tuple
import random
import warnings
import pytest
import numpy as np
import scipy.sparse as sp
import torch_sparse as tsp
import networkx as nx


def compose_graphs_disjoint(*graphs: nx.Graph) -> nx.Graph:
    num_nodes = 0
    reindexed_graphs = []
    for graph in graphs:
        _num_graph_nodes = graph.number_of_nodes()
        reindexed_graphs.append(
            nx.relabel_nodes(
                nx.convert_node_labels_to_integers(graph),
                {i: i + num_nodes for i in range(_num_graph_nodes)},
            )
        )
        num_nodes += _num_graph_nodes

    return nx.compose_all(reindexed_graphs)


def multi_structure_graph(
    num_nodes, create_using: type, weighted: bool = True
) -> nx.Graph:
    """
    Creates a graph with multiple edge case structures. It contains multiple components:
     - one clique
     - one cycle
     - a tree
     - one isolated node without edges
     - one isolated node with a self-loop
    :param num_nodes: Number of nodes in each of the multi-node components
    :param create_using: Networkx base: Graph or DiGraph
    :param weighted: If include uniformly sampled weights
    :return:
    """
    clique = nx.complete_graph(num_nodes, create_using=create_using)
    cycle = nx.cycle_graph(num_nodes, create_using=create_using)
    if float(nx.__version__[:2]) > 3.1:
        tree = nx.random_labeled_tree(num_nodes)
    else:
        tree = nx.random_tree(num_nodes)
    tree = nx.DiGraph([(u, v) for (u, v) in tree.edges() if u < v])
    tree = create_using(tree)
    singles = create_using()
    singles.add_node(0)
    singles.add_node(1)
    singles.add_edge(1, 1)
    graph = compose_graphs_disjoint(clique, cycle, tree, singles)
    if weighted:
        for u, v, w in graph.edges(data=True):
            w["weight"] = 10 * random.random()
    return graph


def graph2adj(graph: nx.Graph) -> sp.spmatrix:
    """Convert a networkx graph to a scipy sparse matrix."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        adj = nx.adjacency_matrix(graph, weight="weight").T
    return adj


@pytest.fixture(scope="class")
def dir_ms_graph() -> nx.Graph:
    """Directed version of the multi-structure graph"""
    return multi_structure_graph(100, nx.DiGraph, weighted=True)


@pytest.fixture(scope="class")
def undir_ms_graph() -> nx.Graph:
    return multi_structure_graph(100, nx.Graph, weighted=True)


def nxgraph2edges_and_weights(graph: nx.Graph):
    edges = []
    weights = []
    for i, j, w in graph.edges(data="weight"):
        edges.append((i, j))
        weights.append(w)
    edges = np.asarray(edges, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    return edges, weights


@pytest.fixture(scope="class")
def dir_ms_adj_mat(dir_ms_graph) -> tsp.SparseTensor:
    """
    A directed adjacency  matrix consisting of four weakly connected components: a clique, a tree, a cycle and an
    isolated node
    :return:
    """
    return tsp.SparseTensor.from_scipy(graph2adj(dir_ms_graph))


@pytest.fixture(scope="class")
def dir_ms_edges_weights(dir_ms_graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    A directed adjacency  matrix consisting of four weakly connected components: a clique, a tree, a cycle and an
    isolated node
    :return:
    """
    return nxgraph2edges_and_weights(dir_ms_graph)


@pytest.fixture(scope="class")
def undir_ms_adj_mat(undir_ms_graph) -> tsp.SparseTensor:
    """
    A directed adjacency  matrix consisting of four weakly connected components: a clique, a tree, a cycle and an
    isolated node
    :return:
    """
    return tsp.SparseTensor.from_scipy(graph2adj(undir_ms_graph))


@pytest.fixture(scope="class")
def undir_ms_edges_weights(undir_ms_graph) -> Tuple[np.ndarray, np.ndarray]:
    """
    A directed adjacency  matrix consisting of four weakly connected components: a clique, a tree, a cycle and an
    isolated node
    :return:
    """
    adj = graph2adj(undir_ms_graph).tocoo()
    edges = np.stack((adj.col, adj.row), axis=1)
    weights = adj.data
    return edges, weights
