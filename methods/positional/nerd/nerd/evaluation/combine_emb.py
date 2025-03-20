#! /usr/bin/python3


import argparse
import random

import gensim

import functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "EMBEDDINGS", nargs=2, help="The embeddings (gensim W2V) to combine"
    )
    ap.add_argument(
        "--function",
        default="concat",
        help="How to combine the embeddings (concat, avg, had, wl1, wl2)",
    )
    ap.add_argument(
        "--save_as", default="combined.txt", help="Where to save the result"
    )
    ap.add_argument(
        "-b", "--binary", action="store_true", help="Read the embedding files as binary"
    )
    args = ap.parse_args()

    funcs = {
        "concat": functions.concat,
        "avg": functions.average,
        "had": functions.hadamard,
        "wl1": functions.weighted_l1,
        "wl2": functions.weighted_l2,
    }
    if args.function not in funcs:
        exit("error: invalid function {}".format(args.function))
    combine_func = funcs[args.function]

    emb_file1, emb_file2 = args.EMBEDDINGS
    print("reading {}...".format(emb_file1))
    emb1 = functions.read_w2v_emb(emb_file1, args.binary)
    print("reading {}...".format(emb_file2))
    emb2 = functions.read_w2v_emb(emb_file2, args.binary)
    assert emb1.index2word == emb2.index2word

    emb_combined = {}
    for node in emb1.vocab:
        emb_combined[node] = combine_func(emb1[node], emb2[node])

    print("saving as {}...".format(args.save_as))
    functions.save_w2v_format(emb_combined, args.save_as)


if __name__ == "__main__":
    main()
