#! /usr/bin/python3


import argparse
import random

import networkx
import numpy
import gensim

import functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("GRAPH", help="A file containing the graph")
    ap.add_argument(
        "-u", "--unweighted", action="store_true", help="Read an unweighted graph"
    )
    ap.add_argument(
        "-d",
        "--delimiter",
        default=" ",
        help="The delimiter in the edge list (default: space)",
    )
    ap.add_argument("LABELS", help="A file containing the labels of each node ID")
    ap.add_argument(
        "-l",
        "--label_format",
        default="cora",
        help="Specify the label format (cora, blogcat)",
    )
    ap.add_argument("AUTH", help="A file containing the auth. (or regular) embeddings")
    ap.add_argument("--hub", help="A file containing the hub embeddings (optional)")
    ap.add_argument(
        "-cv", "--cross_val_k", type=int, default=3, help="Do k-fold cross-validation"
    )
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
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
    args = ap.parse_args()

    if args.hope and args.verse:
        exit("error: --hope and --verse are mutually exclusive")

    if args.label_format not in ("cora", "blogcat"):
        exit('error: only label formats "cora" and "blogcat" are supported')

    print("reading {}...".format(args.GRAPH))
    if args.unweighted:
        orig_graph = networkx.read_edgelist(
            args.GRAPH,
            nodetype=str,
            create_using=networkx.DiGraph(),
            comments=args.comments,
            delimiter=args.delimiter,
        )
    else:
        orig_graph = networkx.read_edgelist(
            args.GRAPH,
            nodetype=str,
            data=(("weight", int),),
            create_using=networkx.DiGraph(),
            comments=args.comments,
            delimiter=args.delimiter,
        )

    if args.hope:
        print("reading {}...".format(args.AUTH))
        auth = functions.read_hope_emb(args.AUTH)

        if args.hub:
            print("reading {}...".format(args.hub))
            hub = functions.read_hope_emb(args.hub)

            # make sure we have all embeddings
            assert len(auth) == len(hub) == orig_graph.number_of_nodes()

            # check if the indices match
            assert auth.keys() == hub.keys()
        else:
            hub = None
    elif args.verse:
        num_nodes, embedding_dim = args.verse
        print("reading {}...".format(args.AUTH))
        auth = functions.read_verse_emb(args.AUTH, num_nodes, embedding_dim)

        if args.hub:
            print("reading {}...".format(args.hub))
            hub = functions.read_verse_emb(args.hub, num_nodes, embedding_dim)

            # make sure we have all embeddings
            assert len(auth) == len(hub) == orig_graph.number_of_nodes()

            # check if the indices match
            assert auth.keys() == hub.keys()
        else:
            hub = None
    else:
        print("reading {}...".format(args.AUTH))
        auth = functions.read_w2v_emb(args.AUTH, args.binary)

        if args.hub:
            print("reading {}...".format(args.hub))
            hub = functions.read_w2v_emb(args.hub, args.binary)

            # make sure we have all embeddings
            assert len(auth.vocab) == len(hub.vocab) == orig_graph.number_of_nodes()

            # check if the indices match
            assert auth.index2word == hub.index2word
        else:
            hub = None

    print("reading {}...".format(args.LABELS))
    if args.label_format == "cora":
        labels, label_list = functions.read_cora_labels(args.LABELS)
    else:
        labels, label_list = functions.read_blogcat_labels(args.LABELS)
    assert len(labels) == orig_graph.number_of_nodes()

    f1_micro, f1_macro = functions.get_f1_cross_val(
        labels, label_list, args.cross_val_k, auth, hub, True
    )
    print("F1 (micro) = {}".format(f1_micro))
    print("F1 (macro) = {}".format(f1_macro))


if __name__ == "__main__":
    main()
