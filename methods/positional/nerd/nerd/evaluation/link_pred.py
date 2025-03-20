#! /usr/bin/python3


import argparse
import random

import networkx
import gensim

import functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ORIG_GRAPH", help="A file containing the original graph")
    ap.add_argument("TESTING_GRAPH", help="A file containing the testing graph")
    ap.add_argument("AUTH", help="A file containing the auth. embeddings")
    ap.add_argument("HUB", help="A file containing the hub embeddings")
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
    )
    ap.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0,
        help="What fraction of the total negative edges should be reversed positive edges at most",
    )
    ap.add_argument(
        "--hope", action="store_true", help="Use the HOPE embedding format (csv)"
    )
    ap.add_argument(
        "--verse",
        nargs=2,
        type=int,
        help="Use the VERSE embedding format (numpy binary). Expects 2 parameters: Number of nodes and embedding size (dimensions)",
    )
    ap.add_argument(
        "--comments",
        default="#",
        help="A string that indicates the start of a line comment in the graph file",
    )
    ap.add_argument("--random_seed", help="Specify a random seed")
    args = ap.parse_args()

    if args.hope and args.verse:
        exit("error: --hope and --verse are mutually exclusive")

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

    if args.hope:
        print("reading {}...".format(args.AUTH))
        auth = functions.read_hope_emb(args.AUTH)

        print("reading {}...".format(args.HUB))
        hub = functions.read_hope_emb(args.HUB)

        # make sure we have all embeddings
        assert len(auth) == len(hub) == orig_graph.number_of_nodes()

        # check if the indices match
        assert auth.keys() == hub.keys()

        score = functions.get_lp_score(
            orig_graph, testing_graph, args.fraction, auth, hub, True
        )
    elif args.verse:
        num_nodes, embedding_dim = args.verse
        print("reading {}...".format(args.AUTH))
        auth = functions.read_verse_emb(args.AUTH, num_nodes, embedding_dim)

        print("reading {}...".format(args.AUTH))
        hub = functions.read_verse_emb(args.HUB, num_nodes, embedding_dim)

        # make sure we have all embeddings
        assert len(auth) == len(hub) == orig_graph.number_of_nodes()

        score = functions.get_lp_score(
            orig_graph, testing_graph, args.fraction, auth, hub, True
        )
    else:
        print("reading {}...".format(args.AUTH))
        auth = functions.read_w2v_emb(args.AUTH, args.binary)

        print("reading {}...".format(args.HUB))
        hub = functions.read_w2v_emb(args.HUB, args.binary)

        # make sure we have all embeddings
        assert len(auth.vocab) == len(hub.vocab) == orig_graph.number_of_nodes()

        score = functions.get_lp_score(
            orig_graph, testing_graph, args.fraction, auth, hub, True
        )
    print("ROC AUC score: {}".format(score))


if __name__ == "__main__":
    main()
