#! /usr/bin/python3


import argparse
import random
import itertools

import networkx
import numpy
import gensim

import functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("GRAPH", help="A file containing the graph")
    ap.add_argument("AUTH", help="A file containing the auth. embeddings")
    ap.add_argument("HUB", help="A file containing the hub embeddings")
    ap.add_argument(
        "NODES", help="A file containing the nodes (created with create_gr_testset.py)"
    )
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
    )
    ap.add_argument(
        "--k_out",
        type=int,
        nargs="+",
        default=[None],
        help="How many neighbors to consider for out-degree accuracy",
    )
    ap.add_argument(
        "--k_in",
        type=int,
        nargs="+",
        default=[None],
        help="How many neighbors to consider for in-degree accuracy",
    )
    ap.add_argument(
        "--comments",
        default="#",
        help="A string that indicates the start of a line comment in the graph file",
    )
    args = ap.parse_args()

    print("reading {}...".format(args.GRAPH))
    orig_graph = networkx.read_edgelist(
        args.GRAPH,
        nodetype=str,
        data=(("weight", int),),
        create_using=networkx.DiGraph(),
        comments=args.comments,
    )

    print("reading {}...".format(args.AUTH))
    auth = functions.read_w2v_emb(args.AUTH, args.binary)

    print("reading {}...".format(args.HUB))
    hub = functions.read_w2v_emb(args.HUB, args.binary)

    # make sure we have all embeddings
    assert len(auth.vocab) == len(hub.vocab) == orig_graph.number_of_nodes()

    # check if the indices match
    assert auth.index2word == hub.index2word

    print("reading {}...".format(args.NODES))
    with open(args.NODES) as hfile:
        nodes = [line.strip() for line in hfile]

    micro_avg_out, macro_avg_out, micro_avg_in, macro_avg_in = functions.get_gr_acc(
        nodes, orig_graph, auth, hub, set(args.k_out), set(args.k_in), True
    )
    print("out: avg. accuracy (micro): {}".format(micro_avg_out))
    print("out: avg. accuracy (macro): {}".format(macro_avg_out))
    print("in: avg. accuracy (micro): {}".format(micro_avg_in))
    print("in: avg. accuracy (macro): {}".format(macro_avg_in))


if __name__ == "__main__":
    main()
