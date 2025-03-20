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
    ap.add_argument("NODES", help="A file containing the nodes")
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
    )
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[100],
        help="How many nearest neighbors to consider",
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

    scores = functions.get_gr_acc_sigmoid(
        nodes, orig_graph, auth, hub, set(args.k), True
    )
    print("scores: {}".format(scores))


if __name__ == "__main__":
    main()
