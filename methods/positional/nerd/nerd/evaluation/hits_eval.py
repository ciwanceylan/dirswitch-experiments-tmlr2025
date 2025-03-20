#! /usr/bin/python3


import argparse
import pickle

import networkx
import gensim

import functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("GRAPH", help="A file containing the graph")
    ap.add_argument("AUTH", help="A file containing the auth. embeddings")
    ap.add_argument("HUB", help="A file containing the hub embeddings")
    ap.add_argument(
        "-hi",
        "--hits_iterations",
        type=int,
        default=1000,
        help="Maximum HITS iterations",
    )
    ap.add_argument(
        "-ht", "--hits_tolerance", type=float, default=1e-5, help="HITS tolerance"
    )
    ap.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.01,
        help="Use only the top hubs and autorities from HITS (between 0 and 1)",
    )
    ap.add_argument("--save_hits", help="Save the HITS results in a file")
    ap.add_argument("--load_hits", help="Load the HITS results from a file")
    ap.add_argument(
        "-ka", "--k_auth", type=int, help="Number of auth. for Spearman (default: all)"
    )
    ap.add_argument(
        "-kh", "--k_hubs", type=int, help="Number of hubs for Spearman (default: all)"
    )
    ap.add_argument(
        "-kna",
        "--k_ndcg_auth",
        type=int,
        help="Number of auth. nodes for NDCG (default: all)",
    )
    ap.add_argument(
        "-knh",
        "--k_ndcg_hubs",
        type=int,
        help="Number of hub nodes for NDCG (default: all)",
    )
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
    )
    args = ap.parse_args()

    # read a weighted directed graph from the source file
    print("reading {}...".format(args.GRAPH))
    orig_graph = networkx.read_edgelist(
        args.GRAPH,
        nodetype=str,
        data=(("weight", int),),
        create_using=networkx.DiGraph(),
    )

    print("reading {}...".format(args.AUTH))
    auth = functions.read_w2v_emb(args.AUTH, args.binary)

    print("reading {}...".format(args.HUB))
    hub = functions.read_w2v_emb(args.HUB, args.binary)

    # check if the indices match
    assert auth.wv.index2word == hub.wv.index2word

    if args.load_hits:
        print("loading results from {}...".format(args.load_hits))
        with open(args.load_hits, "rb") as hfile:
            hits_h, hits_a = pickle.load(hfile)
    else:
        print("running HITS algorithm...")
        hits_h, hits_a = networkx.hits(
            orig_graph, max_iter=args.hits_iterations, tol=args.hits_tolerance
        )
    assert len(hits_h) == len(hits_a)

    if args.save_hits:
        print("saving results to {}...".format(args.save_hits))
        with open(args.save_hits, "wb") as hfile:
            pickle.dump((hits_h, hits_a), hfile)

    k = int(args.fraction * len(hits_h))
    # pylint: disable=E0632
    (
        spear_auth_dot,
        spear_hubs_dot,
        ndcg_auth_dot,
        ndcg_hubs_dot,
        spear_auth_norm,
        spear_hubs_norm,
        ndcg_auth_norm,
        ndcg_hubs_norm,
    ) = functions.get_hits_scores(
        k,
        hits_a,
        hits_h,
        auth,
        hub,
        args.k_auth,
        args.k_hubs,
        args.k_ndcg_auth,
        args.k_ndcg_hubs,
        True,
    )
    print("spearman:")
    print("auth. (dot products): {}".format(spear_auth_dot))
    print("hubs (dot products): {}".format(spear_hubs_dot))
    print("auth. (norm): {}".format(spear_auth_norm))
    print("hubs (norm): {}".format(spear_hubs_norm))
    print("ndcg:")
    print("auth. (dot products): {}".format(ndcg_auth_dot))
    print("hubs (dot products): {}".format(ndcg_hubs_dot))
    print("auth. (norm): {}".format(ndcg_auth_norm))
    print("hubs (norm): {}".format(ndcg_hubs_norm))


if __name__ == "__main__":
    main()
