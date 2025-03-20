#!/usr/bin/env python3

import argparse

import nebtools.data.datasets as datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", type=str, default=None, help="Where to store the data."
    )
    parser.add_argument(
        "--download", action="store_true", default=False, help="Download cloud datasets"
    )
    parser.add_argument(
        "--index-synthetic",
        action="store_true",
        default=False,
        help="Index old synthetic datasets.",
    )
    args = parser.parse_args()
    datasets.index_datasets(
        data_root=args.dataroot,
        download=args.download,
        index_synthetic=args.index_synthetic,
    )
