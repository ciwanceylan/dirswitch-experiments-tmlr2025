import networkx as nx
import torch_sparse as tsp


def get_tsp_element(mat: tsp.SparseTensor, row: int, col: int):
    return mat[row, col].to_dense().item()


def get_nx_degrees(dir_graph: nx.DiGraph, in_degrees: bool, weighted: bool):
    if not in_degrees and weighted:
        return dir_graph.out_degree(weight="weight")
    elif not in_degrees and not weighted:
        return dir_graph.out_degree()
    elif in_degrees and weighted:
        return dir_graph.in_degree(weight="weight")
    elif in_degrees and not weighted:
        return dir_graph.in_degree()
    else:
        return None
