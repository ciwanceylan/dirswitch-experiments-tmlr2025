#! /usr/bin/python3


import argparse
import csv
import random
import os

import networkx

import functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ORIG_GRAPH", help="A file containing the original graph")
    ap.add_argument("TESTING_GRAPH", help="A file containing the testing graph")
    ap.add_argument(
        "AUTH", help="A file that contains the auth. embeddings (gensim W2V)"
    )
    ap.add_argument("HUB", help="A file that contains the hub embeddings (gensim W2V)")
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
    )
    ap.add_argument(
        "-f",
        "--fraction",
        nargs="+",
        type=float,
        default=[0],
        help="What fractions of the total negative edges should be reversed positive edges at most",
    )
    ap.add_argument(
        "--name",
        help="Specify the name of the embedding algorithm (default: embedding file name)",
    )
    ap.add_argument(
        "--comments",
        default="#",
        help="A string that indicates the start of a line comment in the graph file",
    )
    ap.add_argument(
        "--save_as", default="cora_results.csv", help="Where to save the results (csv)"
    )
    ap.add_argument("--random_seed", help="Specify a random seed")
    args = ap.parse_args()

    if args.random_seed:
        random.seed(args.random_seed)

    print("reading {}...".format(args.ORIG_GRAPH))
    orig_graph = networkx.read_edgelist(
        args.ORIG_GRAPH,
        nodetype=str,
        data=(("weight", int),),
        create_using=networkx.DiGraph(),
        comments=args.comments,
    )
    print("reading {}...".format(args.TESTING_GRAPH))
    testing_graph = networkx.read_edgelist(
        args.TESTING_GRAPH,
        nodetype=str,
        data=(("weight", int),),
        create_using=networkx.DiGraph(),
        comments=args.comments,
    )

    print("reading {}...".format(args.AUTH))
    auth = functions.read_w2v_emb(args.AUTH, args.binary)

    print("reading {}...".format(args.HUB))
    hub = functions.read_w2v_emb(args.HUB, args.binary)

    scores = {}
    for f in args.fraction:
        print("processing f = {}...".format(f))
        scores[f] = functions.get_lp_score(orig_graph, testing_graph, f, auth, hub)

    print("saving results in {}...".format(args.save_as))
    name = args.name or os.path.basename(args.W2V_EMB)
    with open(args.save_as, "w") as hfile:
        writer = csv.writer(hfile)
        writer.writerow(["name", "fraction", "score"])
        for f, score in scores.items():
            writer.writerow([name, f, score])


if __name__ == "__main__":
    main()
