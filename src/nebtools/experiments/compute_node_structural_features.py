import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import graph_tool as gt
import graph_tool.centrality as gtcent
import nebtools.utils as nebutils
import nebtools.data.graph as dgraph

import structfeatures.features as stf

STATS_PATH = os.path.join(nebutils.NEB_DATAROOT, "data_stats.json")


def compute_features(graph: dgraph.SimpleGraph):
    features = []
    feature_names = []

    gt_graph = graph.to_gt_graph()

    print(f"Computing degrees...")
    if graph.directed:
        out_degs = gt_graph.degree_property_map("out").a
        in_degs = gt_graph.degree_property_map("in").a
        features.append(out_degs.reshape(-1, 1))
        features.append(in_degs.reshape(-1, 1))
        feature_names += ["out_degree", "in_degree"]
    else:
        degs = gt_graph.degree_property_map("total").a
        features.append(degs.reshape(-1, 1))
        feature_names += ["degree"]

    print(f"Computing lccs...")
    lcc_features, lcc_feature_names = stf.local_clustering_coefficients_features(
        edge_index=graph.edges.T,
        num_nodes=graph.num_nodes,
        as_undirected=not graph.directed,
        weights=None,
        dtype=np.float32,
    )
    features.append(lcc_features)
    feature_names += lcc_feature_names

    print(f"Computing pageranks...")
    pr_centrality = gtcent.pagerank(g=gt_graph)
    features.append(pr_centrality.a.reshape(-1, 1))
    feature_names += ["pagerank_centrality"]

    print(f"Computing betweenness...")
    betweenness_centrality, _ = gtcent.betweenness(g=gt_graph)
    features.append(betweenness_centrality.a.reshape(-1, 1))
    feature_names += ["betweenness_centrality"]

    print(f"Computing closeness...")
    closeness_centrality = gtcent.closeness(g=gt_graph, harmonic=True)
    features.append(closeness_centrality.a.reshape(-1, 1))
    feature_names += ["closeness_centrality"]

    if graph.directed:
        gt_graph.set_reversed(True)

        print(f"Computing pageranks reverse...")
        pr_centrality = gtcent.pagerank(g=gt_graph)
        features.append(pr_centrality.a.reshape(-1, 1))
        feature_names += ["pagerank_centrality_rev"]

        print(f"Computing betweenness reverse...")
        betweenness_centrality, _ = gtcent.betweenness(g=gt_graph)
        features.append(betweenness_centrality.a.reshape(-1, 1))
        feature_names += ["betweenness_centrality_rev"]

        print(f"Computing closeness reverse...")
        closeness_centrality = gtcent.closeness(g=gt_graph, harmonic=True)
        features.append(closeness_centrality.a.reshape(-1, 1))
        feature_names += ["closeness_centrality_rev"]

        gt_graph.set_reversed(False)

    features = np.concatenate(features, axis=1)
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features = (features - features_mean) / np.maximum(features_std, 1e-8)
    features_info = {
        "names": feature_names,
        "mean": features_mean.tolist(),
        "std": features_std.tolist(),
    }
    print(f"Done!\n")

    return features, features_info


def get_cached_struct_features(
    dataset: str, compute_and_save_if_not_exists: bool = True
):
    folder = os.path.join(nebutils.NEB_DATA_STRUCT_FEATURES_ROOT, dataset)
    save_path = os.path.join(folder, dataset + ".npy")
    info_path = os.path.join(folder, "info.json")
    if os.path.exists(save_path):
        features = np.load(save_path)
    elif compute_and_save_if_not_exists:
        os.makedirs(folder, exist_ok=True)
        dataset_spec = dgraph.DatasetSpec(
            dataset,
            force_undirected=False,
            force_unweighted=False,
            rm_node_attributes=False,
            with_self_loops=False,
        )
        data_graph = dgraph.SimpleGraph.from_dataset_spec(
            dataset_spec=dataset_spec, dataroot=nebutils.NEB_DATAROOT
        )
        print(f"Computing  features for graph {dataset}")
        features, features_info = compute_features(data_graph)
        np.save(save_path, features)
        with open(info_path, "w") as fp:
            json.dump(features_info, fp)
    else:
        raise FileNotFoundError(
            f"Could not find any saved features under '{save_path}'"
        )
    return features


def cluster_with_auto_number(features: np.ndarray, seed: int = 234234512):
    labels = []
    scores = []
    num_clusters = np.arange(3, 11, dtype=int)
    for num_cluster in num_clusters:
        model = KMeans(n_clusters=num_cluster, random_state=seed, n_init="auto")
        cluster_labels = model.fit_predict(features)
        labels.append(cluster_labels)
        score = silhouette_score(features, labels=cluster_labels, random_state=seed)
        scores.append(score)
    best_inx = np.argmax(scores)
    num_clusters = num_clusters[best_inx]
    labels = labels[best_inx]
    return labels, num_clusters


def get_cluster_labels(dataset: str, cluster_and_save_if_not_exists: bool = False):
    label_path = os.path.join(
        nebutils.NEB_DATA_STRUCT_FEATURES_ROOT, dataset, "cluster_labels.json"
    )
    if os.path.exists(label_path):
        labels = pd.read_json(label_path, typ="series")

    elif cluster_and_save_if_not_exists:
        features = get_cached_struct_features(
            dataset=dataset, compute_and_save_if_not_exists=True
        )
        labels, num_clusters = cluster_with_auto_number(features)
        labels = pd.Series(labels)
        labels.to_json(label_path)
    else:
        raise FileNotFoundError(
            f"Could not find any cluster labels features under '{label_path}'"
        )
    return labels


if __name__ == "__main__":
    _ = get_cluster_labels("enron_na", cluster_and_save_if_not_exists=True)
    _ = get_cluster_labels("email_eu_core_na", cluster_and_save_if_not_exists=True)
    _ = get_cluster_labels("bitcoin_alpha_na", cluster_and_save_if_not_exists=True)
    _ = get_cluster_labels("bitcoin_trust_na", cluster_and_save_if_not_exists=True)
    _ = get_cluster_labels("polblogs", cluster_and_save_if_not_exists=True)
    _ = get_cluster_labels("pubmed", cluster_and_save_if_not_exists=True)
    _ = get_cluster_labels("subelj_cora", cluster_and_save_if_not_exists=True)
