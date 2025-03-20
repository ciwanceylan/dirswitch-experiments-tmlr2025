#! /usr/bin/python3


import argparse
import json
import os
from collections import namedtuple, OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("SOURCE_FILES", nargs="+", help="All dataset files")
    ap.add_argument("--destination", default=".", help="Where to save the results")
    args = ap.parse_args()

    Item = namedtuple("Item", ["new_id", "title", "venue", "references"])
    table = OrderedDict()

    print("reading data...")
    new_id = 0
    for f in args.SOURCE_FILES:
        with open(f) as hfile:
            for line in hfile:
                obj = json.loads(line)
                orig_id = obj["id"]
                # no duplicates allowed
                assert orig_id not in table
                table[orig_id] = Item(
                    new_id,
                    obj.get("title"),
                    obj.get("venue"),
                    obj.get("references", []),
                )
                new_id += 1

    print("saving data...")
    with (
        open(os.path.join(args.destination, "result_graph.txt"), "w") as result_graph,
        open(
            os.path.join(args.destination, "result_properties.txt"), "w"
        ) as result_properties,
    ):
        result_properties.write("id\ttitle\tvenue\n")
        for item in table.values():
            for orig_id in item.references:
                new_id = table[orig_id].new_id
                # add constant weight (1) for all of the edges
                result_graph.write("{}\t{}\t1\n".format(item.new_id, new_id))
                # sometimes the strings are empty
                title = item.title or "???"
                venue = item.venue or "???"
            result_properties.write("{}\t{}\t{}\n".format(item.new_id, title, venue))


if __name__ == "__main__":
    main()
