#! /usr/bin/python3


import argparse
import os

import networkx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("GRAPH", help="The undirected graph")
    ap.add_argument("TARGET_FILE", help="Where to save the directed graph")
    args = ap.parse_args()

    graph = networkx.read_edgelist(
        args.GRAPH, nodetype=str, data=(("weight", int),), create_using=networkx.Graph()
    )
    di_graph = networkx.DiGraph(graph)
    assert 2 * graph.number_of_edges() == di_graph.number_of_edges()
    networkx.write_weighted_edgelist(di_graph, args.TARGET_FILE)


if __name__ == "__main__":
    main()
