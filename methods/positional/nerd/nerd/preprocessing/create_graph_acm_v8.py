#! /usr/bin/python3


import argparse
import os


def items(file_handle):
    item = {"references": []}
    for line in file_handle:
        ls = line.strip()

        # empty line => done with current item
        if not ls:
            yield item
            item = {"references": []}
        elif ls.startswith("#c"):
            item["venue"] = ls[2:]
        elif ls.startswith("#index"):
            item["index"] = ls[6:]
        elif ls.startswith("#%"):
            item["references"].append(ls[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("SOURCE_FILE", help="The dataset")
    ap.add_argument("--save_as", default="result.txt", help="Where to save the results")
    ap.add_argument(
        "-fv",
        "--filter_venue",
        help="Only include items whose venues match this string",
    )
    args = ap.parse_args()

    with (
        open(args.SOURCE_FILE, encoding="utf-8") as source_file,
        open(args.save_as, "w", encoding="utf-8") as dest_file,
    ):
        for item in items(source_file):
            # we skip an item if the venue does not match or it has no index
            if (
                args.filter_venue
                and not args.filter_venue == item.get("venue")
                or "index" not in item
            ):
                continue
            for ref in item["references"]:
                dest_file.write("{}\t{}\n".format(item["index"], ref))


if __name__ == "__main__":
    main()
