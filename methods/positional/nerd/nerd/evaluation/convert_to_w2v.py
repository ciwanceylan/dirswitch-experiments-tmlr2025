#! /usr/bin/python3


import argparse
import random
import csv
from collections import OrderedDict

import numpy

import functions


def read_hope_emb(emb_file):
    result = OrderedDict()
    with open(emb_file) as hfile:
        for i, row in enumerate(csv.reader(hfile)):
            result[str(i + 1)] = list(map(float, row))
    return result


def read_verse_emb(emb_file, num_nodes, embedding_dim):
    result = OrderedDict()
    # pylint: disable=E1101
    data = numpy.fromfile(emb_file, numpy.float32).reshape(num_nodes, embedding_dim)
    for i, row in enumerate(data):
        result[str(i + 1)] = list(map(float, row))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("EMB_FILE", help="A file containing the embeddings")
    ap.add_argument("EMB_TYPE", help="The type of the embeddings (hope, verse)")
    ap.add_argument(
        "EMB_NUM_NODES", type=int, help="The number of nodes in the embeddings"
    )
    ap.add_argument("EMB_DIM", type=int, help="The dimension of the embeddings")
    ap.add_argument("TARGET_FILE", help="Where to save the result")
    args = ap.parse_args()

    if args.EMB_TYPE == "hope":
        print("reading HOPE embeddings from {}...".format(args.EMB_FILE))
        emb = read_hope_emb(args.EMB_FILE)
    elif args.EMB_TYPE == "verse":
        print("reading VERSE embeddings from {}...".format(args.EMB_FILE))
        emb = read_verse_emb(args.EMB_FILE, args.EMB_NUM_NODES, args.EMB_DIM)

    print("saving w2v embeddings in {}...".format(args.TARGET_FILE))
    functions.save_w2v_format(emb, args.TARGET_FILE)


if __name__ == "__main__":
    main()
