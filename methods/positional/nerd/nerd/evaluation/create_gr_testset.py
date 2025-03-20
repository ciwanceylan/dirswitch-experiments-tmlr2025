#! /usr/bin/python3


import argparse
import random

import networkx
import numpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("GRAPH", help="A file containing the graph")
    ap.add_argument(
        "-f",
        "--fraction",
        type=float,
        help="Sample a fraction of all nodes with out-degree > 0",
    )
    ap.add_argument("--out_deg", action="store_true", help="Sample based on out-degree")
    ap.add_argument(
        "--comments",
        default="#",
        help="A string that indicates the start of a line comment in the graph file",
    )
    ap.add_argument(
        "--save_as", default="gr_nodes.txt", help="Save the nodes in this file"
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

    if args.out_deg:
        sum_out_deg = orig_graph.number_of_edges()
        nodes = []
        prob = []
        for node, out_deg in orig_graph.out_degree():
            if out_deg > 0:
                nodes.append(node)
                prob.append(out_deg / sum_out_deg)

    if args.fraction:
        sample_size = int(args.fraction * orig_graph.number_of_nodes())
        print(
            "sampling {} out of {} nodes...".format(
                sample_size, orig_graph.number_of_nodes()
            )
        )
        if args.out_deg:
            sample = numpy.random.choice(nodes, sample_size, replace=False, p=prob)
        else:
            sample = numpy.random.choice(orig_graph.nodes(), sample_size, replace=False)
    else:
        print("using all {} nodes...".format(len(nodes)))
        sample = list(orig_graph.nodes())

    with open(args.save_as, "w") as hfile:
        hfile.write("\n".join(sample))


if __name__ == "__main__":
    main()
