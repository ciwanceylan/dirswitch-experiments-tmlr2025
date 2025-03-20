import abc
import warnings
import dataclasses as dc
from typing import List, Dict
import time
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
import torch

import nebtools.data.graph as dgraph
import nebtools.experiments.classification as nodeclassification
import nebtools.experiments.pt_sgd_log_reg as ssngnnlrcls
import nebtools.algs.utils as algutils


class SSGNNTrainer(abc.ABC):
    @abc.abstractmethod
    def step(self, step: int):
        raise NotImplementedError

    @abc.abstractmethod
    def get_embeddings(self) -> torch.Tensor:
        raise NotImplementedError

    def get_embeddings_for_graph(self, graph: dgraph.SimpleGraph) -> torch.Tensor:
        raise NotImplementedError


class TestEvalCallback:
    name: str = "abc"

    def evaluate(self, embeddings: torch.Tensor) -> List[Dict]:
        raise NotImplementedError

    def __call__(self, embeddings: torch.Tensor) -> List[Dict]:
        return self.evaluate(embeddings)


class EvalClassificationCallback(TestEvalCallback):
    name: str = "nc"

    def __init__(
        self,
        labels: pd.Series,
        node_labels_type: str,
        pp_modes,
        seed: int,
        test_ratio: float = 0.2,
        y_train_test: tuple = None,
    ):
        self.labels = labels
        self.pp_modes = pp_modes
        self.seed = seed
        self.y_train = self.y_test = None
        if y_train_test is not None:
            self.y_train = labels[y_train_test[0]]
            self.y_test = labels[y_train_test[1]]
        n_splits = int(1.0 / test_ratio)

        if node_labels_type == "multiclass":
            self.evaluator = nodeclassification.MultiClassEvaluator(
                random_state=seed,
                with_train_eval=False,
                n_repeats=3,
                n_splits=n_splits,
                train_ratio=1.0 - test_ratio,
            )
        elif node_labels_type == "binary":
            self.evaluator = nodeclassification.BinaryEvaluator(
                random_state=seed,
                with_train_eval=False,
                n_repeats=3,
                n_splits=n_splits,
                train_ratio=1.0 - test_ratio,
            )
        else:
            raise NotImplementedError(
                f"Evaluation not implemented for '{node_labels_type}'."
            )

        self.model = self.get_classification_model(node_labels_type, random_state=seed)

    @staticmethod
    def get_classification_model(node_labels_type: str, random_state: int):
        C = 100.0

        if node_labels_type == "multilabel":
            model = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    random_state=random_state,
                    multi_class="ovr",
                ),
                n_jobs=-1,
            )
        else:
            model = HistGradientBoostingClassifier(
                class_weight="balanced", random_state=random_state
            )
        return model

    @torch.no_grad()
    def evaluate(self, embeddings: torch.Tensor) -> List[Dict]:
        with warnings.catch_warnings():
            # Ignore convergence warnings caused by identical embedding vectors
            warnings.simplefilter("ignore", category=ConvergenceWarning)

            if self.y_test is None:
                results = nodeclassification.pp_and_cv_evaluate(
                    embeddings=embeddings.cpu().numpy(),
                    model=self.model,
                    pp_modes=self.pp_modes,
                    evaluator=self.evaluator,
                    alg_name="ssgnn",
                    y=self.labels,
                    scores_name="",
                )
            else:
                results = nodeclassification.pp_and_evaluate(
                    embeddings=embeddings.cpu().numpy(),
                    model=self.model,
                    pp_modes=self.pp_modes,
                    evaluator=self.evaluator,
                    alg_name="ssgnn",
                    y_train=self.y_train,
                    y_test=self.y_test,
                    scores_name="",
                )
        results = [dc.asdict(s) for s in results]
        return results


class EvalClassificationPTLRCallback(TestEvalCallback):
    name: str = "nc_ptlr"

    def __init__(
        self,
        labels: pd.Series,
        *,
        seed: int,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        max_epoch: int = 300,
        test_ratio: float = 0.2,
        n_repeats: int = 3,
    ):
        self.labels = labels
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.test_ratio = test_ratio
        self.n_repeats = n_repeats

    def evaluate(self, embeddings: torch.Tensor) -> List[Dict]:
        results = ssngnnlrcls.cv_evaluate(
            embeddings=embeddings.detach(),
            labels=self.labels,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epoch=self.max_epoch,
            test_ratio=self.test_ratio,
            seed=self.seed,
            n_repeats=self.n_repeats,
            mute=True,
        )

        results = [dc.asdict(s) for s in results]
        return results


# class EvalLinkPredCallback(TestEvalCallback):
#
#     def __init__(self, lp_obj: lputils.LinkPredictionObjective):
#         self.lp_obj = lp_obj
#
#     def evaluate(self, embeddings: np.ndarray) -> List[Dict]:
#         lp_data = lputils.emb2lp_data(embeddings, asymmetric_embeddings=True)
#         lp_res = lputils.evaluate_lp_pp(
#             lp_data=lp_data,
#             lp_obj=self.lp_obj, pp_modes=('none',)
#         )
#         return lp_res


def get_evaluation(model_trainer: SSGNNTrainer, eval_cb: TestEvalCallback, epoch: int):
    emb = model_trainer.get_embeddings()
    scores = eval_cb.evaluate(emb)

    for score in scores:
        score["epoch"] = epoch
        if emb is not None:
            score["emb_dim"] = emb.shape[1]
    return scores


def get_features(
    graph: dgraph.SimpleGraph, add_degree: bool, add_lcc: bool, standardize: bool
):
    features = algutils.get_features(
        graph=graph, add_degree=add_degree, add_lcc=add_lcc
    )

    if standardize:
        std = np.std(features, axis=0)
        std[std == 0] = 1
        features = (features - np.mean(features, axis=0)) / std

    return torch.from_numpy(features).to(dtype=torch.float32)
