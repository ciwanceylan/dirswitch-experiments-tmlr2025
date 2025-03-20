#! /usr/bin/python3


import argparse
import random

import networkx

import functions


def save_labels(labels, file_name):
    with open(file_name, "w", encoding="utf-8") as hfile:
        for node, node_labels in labels:
            hfile.write("{} {}\n".format(node, ",".join(node_labels)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("LABELS", help="A file containing the labels of each node ID")
    ap.add_argument(
        "-l",
        "--label_format",
        default="cora",
        help="Specify the label format (cora, blogcat)",
    )
    ap.add_argument(
        "--comments",
        default="#",
        help="A string that indicates the start of a line comment in the graph file",
    )
    ap.add_argument(
        "-s",
        "--split",
        type=float,
        default=0.8,
        help="What fraction of labels to use for training",
    )
    ap.add_argument("--no_dev", action="store_true", help="Do not create a dev set")
    ap.add_argument(
        "--save_training_labels",
        default="train_labels.txt",
        help="Save the training labels in this file",
    )
    ap.add_argument(
        "--save_testing_labels",
        default="test_labels.txt",
        help="Save the testing labels in this file",
    )
    ap.add_argument(
        "--save_dev_labels",
        default="dev_labels.txt",
        help="Save the dev labels in this file",
    )
    args = ap.parse_args()

    print("reading {}...".format(args.LABELS))
    if args.label_format == "cora":
        labels, _ = functions.read_cora_labels(args.LABELS)
    else:
        labels, _ = functions.read_blogcat_labels(args.LABELS)

    num_training_labels = int(len(labels) * args.split)
    num_testing_dev_labels = int((len(labels) - num_training_labels) / 2)

    # due to rounding we might not assign all labels to a set.
    # if there are some labels left we assign them to the training set
    num_training_labels += (
        len(labels) - num_training_labels - 2 * num_testing_dev_labels
    )

    label_list = list(labels.items())
    random.shuffle(label_list)
    training_labels = label_list[:num_training_labels]

    if args.no_dev:
        testing_labels = label_list[
            num_training_labels : num_training_labels + 2 * num_testing_dev_labels
        ]
        assert len(labels) == len(training_labels) + len(testing_labels)
        save_labels(training_labels, args.save_training_labels)
        save_labels(testing_labels, args.save_testing_labels)
    else:
        testing_labels = label_list[
            num_training_labels : num_training_labels + num_testing_dev_labels
        ]
        dev_labels = label_list[
            num_training_labels + num_testing_dev_labels : num_training_labels
            + 2 * num_testing_dev_labels
        ]
        assert len(labels) == len(training_labels) + len(testing_labels) + len(
            dev_labels
        )
        save_labels(training_labels, args.save_training_labels)
        save_labels(testing_labels, args.save_testing_labels)
        save_labels(dev_labels, args.save_dev_labels)


if __name__ == "__main__":
    main()
