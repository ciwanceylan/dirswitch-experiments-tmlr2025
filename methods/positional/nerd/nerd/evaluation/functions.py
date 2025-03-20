import random
import math
import csv
from collections import defaultdict

import numpy
from scipy.stats import spearmanr, hmean
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as sk_shuffle
import gensim
from sortedcontainers import SortedListWithKey


# HITS evaluation
def ndcg(ranking, relevances, k):
    dcg = 0
    rels_sorted = sorted(relevances, reverse=True)
    idcg = 0
    for i in range(1, k + 1):
        rel = max(0, relevances[ranking[i - 1]])
        dcg += rel / math.log2(i + 1)
        idcg += max(0, rels_sorted[i - 1]) / math.log2(i + 1)
    return dcg / idcg


def get_hits_scores(
    k,
    hits_a,
    hits_h,
    auth,
    hub,
    k_auth=None,
    k_hubs=None,
    k_ndcg_auth=None,
    k_ndcg_hubs=None,
    verbose=False,
):
    if verbose:
        print("using the top-{} hubs and authorities...".format(k))

    # sort by score
    hits_top_h = sorted(hits_h.items(), key=lambda x: x[1], reverse=True)[:k]
    hits_top_a = sorted(hits_a.items(), key=lambda x: x[1], reverse=True)[:k]

    top_emb_h = []
    top_emb_a = []
    for h, a in zip(hits_top_h, hits_top_a):
        h_id, _ = h
        a_id, _ = a
        top_emb_h.append(hub[h_id])
        top_emb_a.append(auth[a_id])

    top_emb_h_mat = numpy.asarray(list(top_emb_h))
    top_emb_a_mat = numpy.asarray(list(top_emb_a))

    hits_top_h_scores = list(map(lambda x: x[1], hits_top_h))
    hits_top_a_scores = list(map(lambda x: x[1], hits_top_a))

    # spear_auth_dot, spear_hubs_dot, ndcg_auth_dot, ndcg_hubs_dot, spear_auth_norm, spear_hubs_norm, ndcg_auth_norm, ndcg_hubs_norm
    return get_hits_scores_dot(
        hits_top_a_scores,
        hits_top_h_scores,
        top_emb_a_mat,
        top_emb_h_mat,
        k_auth,
        k_hubs,
        k_ndcg_auth,
        k_ndcg_hubs,
    ) + get_hits_scores_norm(
        hits_top_a_scores,
        hits_top_h_scores,
        top_emb_a_mat,
        top_emb_h_mat,
        k_auth,
        k_hubs,
        k_ndcg_auth,
        k_ndcg_hubs,
    )


def get_hits_scores_dot(
    hits_top_a_scores,
    hits_top_h_scores,
    top_emb_a_mat,
    top_emb_h_mat,
    k_auth=None,
    k_hubs=None,
    k_ndcg_auth=None,
    k_ndcg_hubs=None,
):
    # hubs
    hub_prod = numpy.matmul(top_emb_h_mat, numpy.transpose(top_emb_a_mat))
    if k_hubs is not None:
        hub_scores = numpy.sum(numpy.clip(hub_prod, 0, None)[:k_hubs], axis=1)
        hits_top_h_scores = hits_top_h_scores[:k_hubs]
    else:
        hub_scores = numpy.sum(numpy.clip(hub_prod, 0, None), axis=1)
    spearman_hubs = spearmanr(hub_scores, hits_top_h_scores)

    hub_ranking = [
        i for i, _ in sorted(enumerate(hub_scores), key=lambda x: x[1], reverse=True)
    ]
    if k_ndcg_hubs is None:
        k_ndcg_hubs = len(hub_ranking)
    ndcg_hubs = ndcg(hub_ranking, hits_top_h_scores, k_ndcg_hubs)

    # auth.
    auth_prod = numpy.matmul(top_emb_a_mat, numpy.transpose(top_emb_h_mat))
    if k_auth is not None:
        auth_scores = numpy.sum(numpy.clip(auth_prod, 0, None)[:k_auth], axis=1)
        hits_top_a_scores = hits_top_a_scores[:k_auth]
    else:
        auth_scores = numpy.sum(numpy.clip(auth_prod, 0, None), axis=1)
    spearman_auth = spearmanr(auth_scores, hits_top_a_scores)

    auth_ranking = [
        i for i, _ in sorted(enumerate(auth_scores), key=lambda x: x[1], reverse=True)
    ]
    if k_ndcg_auth is None:
        k_ndcg_auth = len(auth_ranking)
    ndcg_auth = ndcg(auth_ranking, hits_top_a_scores, k_ndcg_auth)

    return spearman_auth, spearman_hubs, ndcg_auth, ndcg_hubs


def get_hits_scores_norm(
    hits_top_a_scores,
    hits_top_h_scores,
    top_emb_a_mat,
    top_emb_h_mat,
    k_auth=None,
    k_hubs=None,
    k_ndcg_auth=None,
    k_ndcg_hubs=None,
):
    # hubs
    if k_hubs is not None:
        hub_scores = numpy.linalg.norm(top_emb_h_mat[:k_hubs], axis=1)
        hits_top_h_scores = hits_top_h_scores[:k_hubs]
    else:
        hub_scores = numpy.linalg.norm(top_emb_h_mat, axis=1)
    spearman_hubs = spearmanr(hub_scores, hits_top_h_scores)

    hub_ranking = [
        i for i, _ in sorted(enumerate(hub_scores), key=lambda x: x[1], reverse=True)
    ]
    if k_ndcg_hubs is None:
        k_ndcg_hubs = len(hub_ranking)
    ndcg_hubs = ndcg(hub_ranking, hits_top_h_scores, k_ndcg_hubs)

    # auth
    if k_auth is not None:
        auth_scores = numpy.linalg.norm(top_emb_a_mat[:k_auth], axis=1)
        hits_top_a_scores = hits_top_a_scores[:k_auth]
    else:
        auth_scores = numpy.linalg.norm(top_emb_a_mat, axis=1)
    spearman_auth = spearmanr(auth_scores, hits_top_a_scores)

    auth_ranking = [
        i for i, _ in sorted(enumerate(auth_scores), key=lambda x: x[1], reverse=True)
    ]
    if k_ndcg_auth is None:
        k_ndcg_auth = len(auth_ranking)
    ndcg_auth = ndcg(auth_ranking, hits_top_a_scores, k_ndcg_auth)

    return spearman_auth, spearman_hubs, ndcg_auth, ndcg_hubs


# link prediction
def sample_neg_edges(graph, pos_edges, number_of_edges, fraction, verbose=False):
    # fraction: how many of the total negative edges should be reversed positive edges at most
    result = set()
    current = 0
    max_reversed = int(fraction * number_of_edges)
    for a, b in pos_edges:
        if verbose:
            progress = int(100 * current / number_of_edges)
            print("{}% [{}/{}]".format(progress, current, number_of_edges), end="\r")

        if current < max_reversed and not graph.has_edge(b, a):
            result.add((b, a))
            current += 1

    node_list = list(graph.nodes())
    while current < number_of_edges:
        if verbose:
            progress = int(100 * current / number_of_edges)
            print("{}% [{}/{}]".format(progress, current, number_of_edges), end="\r")

        node1 = random.choice(node_list)
        node2 = random.choice(node_list)
        if not graph.has_edge(node1, node2):
            result.add((node1, node2))
        current = len(result)
    return list(result)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_lp_score(orig_graph, graph, fraction, auth, hub, verbose=False):
    if verbose:
        print("sampling negative edges...")
    pos_edges = list(graph.edges())
    neg_edges = sample_neg_edges(
        orig_graph, pos_edges, graph.number_of_edges(), fraction, verbose
    )

    # make sure we have the correct number of edges
    assert len(pos_edges) == len(neg_edges)
    if verbose:
        print("sampled {} pos. and neg. edges".format(len(pos_edges)))
        print("calculating dot products...")
    __map_func = lambda e: sigmoid(numpy.dot(hub[e[0]], auth[e[1]]))
    x = list(map(__map_func, neg_edges + pos_edges))
    y = [0] * len(neg_edges) + [1] * len(pos_edges)
    return metrics.roc_auc_score(y, x)


# ML classification
def read_cora_labels(labels_file):
    labels = {}
    all_labels = set()
    with open(labels_file, encoding="utf-8") as hfile:
        for i, line in enumerate(hfile):
            # omit the 1st and last item in the list as they are always empty
            node_labels = set(line.strip().split("/")[1:-1])
            labels[str(i + 1)] = node_labels
            all_labels.update(node_labels)
    return labels, all_labels


def read_blogcat_labels(labels_file):
    labels = defaultdict(set)
    all_labels = set()
    with open(labels_file, encoding="utf-8") as hfile:
        for line in hfile:
            node, label = line.strip().split(",")
            labels[node].add(label)
            all_labels.add(label)
    return labels, all_labels


def read_ml_class_labels(labels_file):
    labels = {}
    with open(labels_file, encoding="utf-8") as hfile:
        for line in hfile:
            node, node_labels = line.split(" ")
            labels[node] = set(node_labels.split(","))
    return labels


def get_node_repr(node, auth_emb, hub_emb=None):
    # simply join both embeddings
    if hub_emb is None:
        return auth_emb[node]
    else:
        return list(auth_emb[node]) + list(hub_emb[node])


def get_label_repr(node_labels, label_list):
    # return a binary vector where each index corresponds to a label
    result = []
    for label in label_list:
        if label in node_labels:
            result.append(1)
        else:
            result.append(0)
    return result


def __get_repr(labels, label_list, auth, hub, shuffle=False):
    # sort all labels so the encodings are consistent
    label_list_sorted = sorted(label_list)

    X = []
    y = []
    for node, node_labels in labels.items():
        if hub is not None:
            X.append(get_node_repr(node, auth, hub))
        else:
            X.append(get_node_repr(node, auth))
        y.append(get_label_repr(node_labels, label_list_sorted))
    if shuffle:
        return sk_shuffle(numpy.asarray(X), numpy.asarray(y))
    return numpy.asarray(X), numpy.asarray(y)


def __get_f1(predictions, y, number_of_labels):
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = numpy.argsort(predictions, axis=1)

    # the true number of labels for each node
    num_labels = numpy.sum(y, axis=1)

    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-num:].tolist())

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(range(number_of_labels)).fit_transform(
        pred_reshaped
    )
    f1_micro = f1_score(y, pred_transformed, average="micro")
    f1_macro = f1_score(y, pred_transformed, average="macro")
    return f1_micro, f1_macro


def get_f1_cross_val(labels, label_list, cv, auth, hub, verbose=False):
    # workaround to predict probabilities during cross-validation
    # "method='predict_proba'" does not seem to work
    class ovrc_prob(OneVsRestClassifier):
        def predict(self, X):
            return self.predict_proba(X)

    if verbose:
        print("transforming inputs...")
    X, y = __get_repr(labels, label_list, auth, hub, True)

    if verbose:
        print("shape of X: {}".format(X.shape))
        print("shape of y: {}".format(y.shape))
        print("running {}-fold cross-validation...".format(cv))
    ovrc = ovrc_prob(LogisticRegression())
    pred = cross_val_predict(ovrc, X, y, cv=cv)
    return __get_f1(pred, y, len(label_list))


def get_f1(train_labels, labels, label_list, auth, hub, verbose=False):
    if verbose:
        print("transforming inputs...")
    # these labels are already shuffled
    X_train, y_train = __get_repr(train_labels, label_list, auth, hub, False)
    X, y = __get_repr(labels, label_list, auth, hub, False)

    if verbose:
        print("shape of X_train: {}".format(X_train.shape))
        print("shape of y_train: {}".format(y_train.shape))
        print("shape of X: {}".format(X.shape))
        print("shape of y: {}".format(y.shape))
        print("fitting classifier...")
    ovrc = OneVsRestClassifier(LogisticRegression())
    ovrc.fit(X_train, y_train)

    if verbose:
        print("evaluating...")
    pred = ovrc.predict_proba(X)
    return __get_f1(pred, y, len(label_list))


# graph reconstruction
def cosine_sim(a, b):
    return numpy.dot(a, b) / numpy.linalg.norm(a) / numpy.linalg.norm(b)


def get_gr_acc(
    nodes, orig_graph, auth, hub, k_out_set=None, k_in_set=None, verbose=False
):
    micro_accs_out = defaultdict(list)
    total_true_pos_out = defaultdict(lambda: 0)
    total_out_deg = 0

    micro_accs_in = defaultdict(list)
    total_true_pos_in = defaultdict(lambda: 0)
    total_in_deg = 0

    if k_out_set is None:
        k_out_set = set()
    if k_in_set is None:
        k_in_set = set()
    # if k is None, we divide by the degree
    k_out_set.add(None)
    k_in_set.add(None)

    current = 1
    num_nodes = len(nodes)
    for node in nodes:
        if verbose:
            progress = int(100 * current / num_nodes)
            print("{}% [{}/{}]".format(progress, current, num_nodes), end="\r")
            current += 1

        out_deg = orig_graph.out_degree(node)
        total_out_deg += out_deg
        if out_deg > 0:
            true_neighbors_out = set(orig_graph.successors(node))
            for k_out in k_out_set:
                nearest_out = set(
                    [
                        n
                        for n, _ in auth.similar_by_vector(
                            hub[node], topn=k_out or out_deg
                        )
                    ][:k_out]
                )
                true_pos_out = len(true_neighbors_out & nearest_out)
                total_true_pos_out[k_out] += true_pos_out
                micro_accs_out[k_out].append(true_pos_out / (k_out or out_deg))

        in_deg = orig_graph.in_degree(node)
        total_in_deg += in_deg
        if in_deg > 0:
            true_neighbors_in = set(orig_graph.predecessors(node))
            for k_in in k_in_set:
                nearest_in = set(
                    [
                        n
                        for n, _ in hub.similar_by_vector(
                            auth[node], topn=k_in or in_deg
                        )
                    ][:k_in]
                )
                true_pos_in = len(true_neighbors_in & nearest_in)
                total_true_pos_in[k_in] += true_pos_in
                micro_accs_in[k_in].append(true_pos_in / (k_in or in_deg))

    micro_avg_out = {}
    macro_avg_out = {}
    for k, accs in micro_accs_out.items():
        micro_avg_out[k] = numpy.average(accs)
        if k is None:
            macro_avg_out[k] = total_true_pos_out[k] / total_out_deg
        else:
            macro_avg_out[k] = total_true_pos_out[k] / (k * len(accs))

    micro_avg_in = {}
    macro_avg_in = {}
    for k, accs in micro_accs_in.items():
        micro_avg_in[k] = numpy.average(accs)
        if k is None:
            macro_avg_in[k] = total_true_pos_in[k] / total_in_deg
        else:
            macro_avg_in[k] = total_true_pos_in[k] / (k * len(accs))

    return micro_avg_out, macro_avg_out, micro_avg_in, macro_avg_in


class Similarities(object):
    def __init__(self, k):
        self.k = k
        # sort by negative similarity -> highest first, lowest gets removed
        self.top_k_sims = SortedListWithKey(key=lambda x: -x[2])

    def add(self, node1, node2, sim, prob=None):
        self.top_k_sims.add((node1, node2, sim, prob))
        if len(self.top_k_sims) > self.k:
            self.top_k_sims.pop()

    def get_true_pos_prob(self, k, orig_graph):
        THRESH = 0.51
        count = 0
        for i in range(k):
            _, _, _, prob = self.top_k_sims[i]
            if prob > THRESH:
                count += 1
        return count

    def get_true_pos(self, k, orig_graph):
        count = 0
        for i in range(k):
            node1, node2, _, _ = self.top_k_sims[i]
            if orig_graph.has_edge(node1, node2):
                count += 1
        return count

    def precision(self, k, orig_graph):
        return self.get_true_pos(k, orig_graph) / k

    def weighted_precision(self, k, orig_graph):
        weights = 0
        total_weights = 0
        for i in range(k):
            node1, node2, sim, _ = self.top_k_sims[i]
            if orig_graph.has_edge(node1, node2):
                weights += sim
            total_weights += sim
        if total_weights == 0:
            return 0
        return weights / total_weights


def get_gr_acc_sigmoid(nodes, orig_graph, auth, hub, k_set, verbose=False):
    micro_accs_out = defaultdict(list)
    micro_accs_in = defaultdict(list)

    current = 1
    num_nodes = len(nodes)
    for node in nodes:
        if verbose:
            progress = int(100 * current / num_nodes)
            print("{}% [{}/{}]".format(progress, current, num_nodes), end="\r")
            current += 1

        sims_out = Similarities(max(k_set))
        nearest_out = set(
            [n for n, _ in auth.similar_by_vector(hub[node], topn=max(k_set))]
        )
        for neighbor in nearest_out:
            sim = numpy.dot(hub[node], auth[neighbor])
            sims_out.add(node, neighbor, sim, sigmoid(sim))
        for k_out in k_set:
            true_pos_out = sims_out.get_true_pos(k_out, orig_graph)
            if orig_graph.out_degree(node) > 0:
                micro_accs_out[k_out].append(true_pos_out / k_out)
            elif sims_out.get_true_pos_prob(k_out, orig_graph) == 0:
                micro_accs_out[k_out].append(1)
            else:
                micro_accs_out[k_out].append(0)

        sims_in = Similarities(max(k_set))
        nearest_in = set(
            [n for n, _ in hub.similar_by_vector(auth[node], topn=max(k_set))]
        )
        for neighbor in nearest_in:
            sim = numpy.dot(hub[neighbor], auth[node])
            sims_in.add(neighbor, node, sim, sigmoid(sim))
        for k_in in k_set:
            true_pos_in = sims_in.get_true_pos(k_in, orig_graph)
            if orig_graph.in_degree(node) > 0:
                micro_accs_in[k_in].append(true_pos_in / k_in)
            elif sims_in.get_true_pos_prob(k_in, orig_graph) == 0:
                micro_accs_in[k_in].append(1)
            else:
                micro_accs_in[k_in].append(0)

    scores = {}
    EPS = 1e-5
    for k in k_set:
        micro_score = [
            hmean((i + EPS, o + EPS))
            for i, o in zip(micro_accs_in[k], micro_accs_out[k])
        ]
        scores[k] = numpy.average(micro_score)
    return scores


# misc
def read_w2v_emb(file_path, binary):
    return gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=binary).wv


def read_hope_emb(emb_file):
    with open(emb_file) as hfile:
        return {
            str(i + 1): list(map(float, row)) for i, row in enumerate(csv.reader(hfile))
        }


def read_verse_emb(emb_file, num_nodes, embedding_dim):
    # pylint: disable=E1101
    data = numpy.fromfile(emb_file, numpy.float32).reshape(num_nodes, embedding_dim)
    return {str(i + 1): list(map(float, row)) for i, row in enumerate(data)}


# combining embeddings
def concat(vec1, vec2):
    return numpy.concatenate([vec1, vec2])


def average(vec1, vec2):
    return (vec1 + vec2) / 2


def hadamard(vec1, vec2):
    return numpy.multiply(vec1, vec2)


def weighted_l1(vec1, vec2):
    return numpy.abs(vec1 - vec2)


def weighted_l2(vec1, vec2):
    return numpy.abs(numpy.square(vec1 - vec2))


def save_w2v_format(embeddings, file_path):
    num_nodes = len(embeddings)
    emb_dim = len(next(iter(embeddings.values())))
    with open(file_path, "w") as hfile:
        hfile.write("{} {}\n".format(num_nodes, emb_dim))
        for node, embedding in embeddings.items():
            hfile.write("{} ".format(node))
            hfile.write(" ".join(map(str, embedding)))
            hfile.write("\n")
