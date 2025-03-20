#! /usr/bin/python3


import argparse
import csv
from collections import defaultdict
import random
import math

from matplotlib import pyplot
from matplotlib import patheffects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("RESULTS", nargs="+", help="The csv files")
    ap.add_argument("--x", required=True, help="What csv column to use as x-axis")
    ap.add_argument("--y", required=True, help="What csv column to use as y-axis")
    ap.add_argument("--save_as", help="Where to save the plot")
    ap.add_argument("--xlabel", help="Override the label of the x-axis")
    ap.add_argument("--ylabel", help="Override the label of the y-axis")
    ap.add_argument(
        "--titles", nargs="+", help="Specify titles in the same order as the csv files"
    )
    ap.add_argument(
        "--log_scale", action="store_true", help="Use a logarithmic x-scale"
    )
    ap.add_argument("--max_x", type=float, help="Do not plot x values larger than this")
    ap.add_argument("--figsize", nargs=2, type=int, help="Figure size")
    ap.add_argument("--bottomadjust", type=float, help="Adjust the bottom of the plots")
    args = ap.parse_args()

    if args.titles and len(args.titles) != len(args.RESULTS):
        exit("error: must specify the same number of csv files and titles")

    MARKERS = {
        "NERD": "8",
        "DeepWalk": "s",
        "HOPE": "v",
        "VERSE": "^",
        "LINE": "<",
        "LINE2": "<",
        "Node2Vec": ">",
        "APP": "p",
    }

    COLORS = {
        "NERD": "#66c2a5",
        "DeepWalk": "#fc8d62",
        "HOPE": "#8da0cb",
        "VERSE": "#e78ac3",
        "LINE": "#a6d854",
        "LINE2": "#a6d854",
        "Node2Vec": "#ffd92f",
        "APP": "#e5c494",
    }

    pyplot.style.use("ggplot")
    pyplot.rc("font", weight="bold", size=16)

    data = []
    for csvfile in args.RESULTS:
        res = defaultdict(list)
        with open(csvfile) as hfile:
            for row in csv.DictReader(hfile):
                x = float(row[args.x])
                y = float(row[args.y])
                if not args.max_x or x <= args.max_x:
                    res[row["Approach"]].append((x, y))
        data.append(res)

    fig = pyplot.figure(figsize=args.figsize)
    for i, res in enumerate(data):
        ax = fig.add_subplot(1, len(data), i + 1)
        for alg, results in sorted(res.items()):
            x, y = zip(*results)
            pyplot.plot(
                x,
                y,
                label=alg,
                marker=MARKERS.get(alg),
                color=COLORS.get(alg),
                lw=4,
                markersize=10,
                path_effects=[
                    patheffects.Stroke(linewidth=4, foreground="k"),
                    patheffects.Normal(),
                ],
            )

        pyplot.xlabel(args.xlabel or args.x)
        pyplot.ylabel(args.ylabel or args.y)
        if args.log_scale:
            ax.set_xscale("log")

        if args.titles:
            pyplot.title(args.titles[i])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    pyplot.tight_layout()
    if args.bottomadjust:
        fig.subplots_adjust(bottom=args.bottomadjust)

    if args.save_as:
        pyplot.savefig(args.save_as)
    else:
        pyplot.show()


if __name__ == "__main__":
    main()
