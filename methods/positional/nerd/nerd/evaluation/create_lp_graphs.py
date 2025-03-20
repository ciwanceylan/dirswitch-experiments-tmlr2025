#! /usr/bin/python3


import argparse
import random
from collections import defaultdict

import networkx
import numpy


def get_graphs(orig_graph, num_training_edges):
    counts = defaultdict(lambda: 0)
    for a, b in orig_graph.edges():
        counts[a] += 1
        counts[b] += 1

    print("sampling edges...")
    rest_edges = set()
    training_edges = set(orig_graph.edges())
    while len(training_edges) > num_training_edges:
        # we only remove edges if the corresponding nodes have other edges
        # if not all edges in the sample could be remved we repeat this step
        for a, b in random.sample(
            training_edges, len(training_edges) - num_training_edges
        ):
            if counts[a] > 1 and counts[b] > 1:
                training_edges.remove((a, b))
                rest_edges.add((a, b))
                print(len(training_edges), end="\r")
                counts[a] -= 1
                counts[b] -= 1

    # all nodes should have at least one edge
    assert all(c > 0 for c in counts.values())

    print("creating graphs...")
    training_graph = orig_graph.copy()
    training_graph.remove_edges_from(rest_edges)
    assert training_graph.number_of_edges() == num_training_edges
    assert training_graph.number_of_nodes() == orig_graph.number_of_nodes()

    rest_graph = orig_graph.copy()
    rest_graph.remove_edges_from(training_edges)

    return training_graph, rest_graph


def split_graph(graph, ratio):
    num_edges = int(ratio * graph.number_of_edges())
    edges = random.sample(graph.edges(), num_edges)

    g1 = graph.copy()
    g2 = graph.copy()
    g1.remove_edges_from(edges)
    g2.remove_edges_from(g1.edges())

    return g1, g2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("GRAPH", help="A file containing the graph")
    ap.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.5,
        help="What fraction of edges to use for training",
    )
    ap.add_argument("--no_dev", action="store_true", help="Do not create a dev set")
    ap.add_argument(
        "--comments",
        default="#",
        help="A string that indicates the start of a line comment in the graph file",
    )
    ap.add_argument(
        "--save_training_graph",
        default="train_graph.txt",
        help="Save the training graph in this file",
    )
    ap.add_argument(
        "--save_testing_graph",
        default="test_graph.txt",
        help="Save the testing graph in this file",
    )
    ap.add_argument(
        "--save_dev_graph",
        default="dev_graph.txt",
        help="Save the dev graph in this file",
    )
    args = ap.parse_args()

    # read a weighted directed graph from the source file
    print("reading {}...".format(args.GRAPH))
    orig_graph = networkx.read_edgelist(
        args.GRAPH,
        nodetype=str,
        data=(("weight", int),),
        create_using=networkx.DiGraph(),
        comments=args.comments,
    )

    num_training_edges = int(args.fraction * orig_graph.number_of_edges())
    training_graph, rest_graph = get_graphs(orig_graph, num_training_edges)
    assert (
        rest_graph.number_of_edges() + training_graph.number_of_edges()
        == orig_graph.number_of_edges()
    )

    if args.no_dev:
        networkx.write_weighted_edgelist(training_graph, args.save_training_graph)
        networkx.write_weighted_edgelist(rest_graph, args.save_testing_graph)
    else:
        # split the rest into testing and dev graphs
        testing_graph, dev_graph = split_graph(rest_graph, 0.5)
        assert (
            testing_graph.number_of_edges()
            + dev_graph.number_of_edges()
            + training_graph.number_of_edges()
            == orig_graph.number_of_edges()
        )

        networkx.write_weighted_edgelist(training_graph, args.save_training_graph)
        networkx.write_weighted_edgelist(testing_graph, args.save_testing_graph)
        networkx.write_weighted_edgelist(dev_graph, args.save_dev_graph)


if __name__ == "__main__":
    main()
