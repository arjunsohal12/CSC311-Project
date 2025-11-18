# pred.py
import json
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from util.dataset_reader import DataReader


def _to_int_list(data):
    return [int(v) for v in data]


@unique
class SplitType(IntEnum):
    numerical = 0
    categorical = 1


@dataclass
class Node:
    left: int
    right: int
    parent: int
    split_idx: int
    split_cond: float
    default_left: bool
    split_type: SplitType
    categories: List[int]
    base_weight: float
    loss_chg: float
    sum_hess: float


class Tree:
    def __init__(self, tree_id: int, nodes: List[Node]) -> None:
        self.tree_id = tree_id
        self.nodes = nodes

    def is_leaf(self, node_id: int) -> bool:
        return self.nodes[node_id].left == -1

    def is_deleted(self, node_id: int) -> bool:
        # XGBoost uses max uint32 for deleted nodes
        return self.nodes[node_id].split_idx == np.iinfo(np.uint32).max

    def left_child(self, node_id: int) -> int:
        return self.nodes[node_id].left

    def right_child(self, node_id: int) -> int:
        return self.nodes[node_id].right

    def split_index(self, node_id: int) -> int:
        return self.nodes[node_id].split_idx

    def split_condition(self, node_id: int) -> float:
        return self.nodes[node_id].split_cond

    def is_categorical(self, node_id: int) -> bool:
        return self.nodes[node_id].split_type == SplitType.categorical

    def categories(self, node_id: int) -> List[int]:
        return self.nodes[node_id].categories

    def default_left(self, node_id: int) -> bool:
        return self.nodes[node_id].default_left

    def predict_one(self, x_row: np.ndarray) -> float:
        node_id = 0
        while True:
            if self.is_leaf(node_id) or self.is_deleted(node_id):
                return float(self.split_condition(node_id))

            split_idx = self.split_index(node_id)
            split_cond = float(self.split_condition(node_id))
            val = x_row[split_idx]

            if np.isnan(val):
                go_left = self.default_left(node_id)
            else:
                if self.is_categorical(node_id):
                    cats = self.categories(node_id)
                    go_left = int(val) in cats
                else:
                    go_left = float(val) < split_cond

            node_id = self.left_child(node_id) if go_left else self.right_child(node_id)


class ParsedModel:
    def __init__(self, model: Dict[str, Any]) -> None:
        learner = model["learner"]
        self.learner_model_shape = learner["learner_model_param"]
        self.num_output_group = int(self.learner_model_shape["num_class"])
        self.num_feature = int(self.learner_model_shape["num_feature"])
        # base_score stored like "[0.36,0.31,0.31]"
        self.base_score: List[float] = json.loads(self.learner_model_shape["base_score"])

        gbm = learner["gradient_booster"]["model"]
        self.tree_info = gbm["tree_info"]  # which class each tree belongs to

        model_shape = gbm["gbtree_model_param"]
        self.num_trees = int(model_shape["num_trees"])

        j_trees = gbm["trees"]

        trees: List[Tree] = []

        for i in range(self.num_trees):
            tree = j_trees[i]
            tree_id = int(tree["id"])
            assert tree_id == i

            # structure arrays (cast to ints where appropriate)
            left_children = _to_int_list(tree["left_children"])
            right_children = _to_int_list(tree["right_children"])
            parents = _to_int_list(tree["parents"])
            split_conditions = [float(v) for v in tree["split_conditions"]]
            split_indices = _to_int_list(tree["split_indices"])
            default_left = _to_int_list(tree["default_left"])
            split_types = _to_int_list(tree["split_type"])

            # categorical CSR-style storage
            cat_segments = _to_int_list(tree["categories_segments"])
            cat_sizes = _to_int_list(tree["categories_sizes"])
            cat_nodes = _to_int_list(tree["categories_nodes"])
            cats = tree["categories"]

            node_categories: List[List[int]] = []
            cat_cnt = 0
            last_cat_node = cat_nodes[cat_cnt] if cat_nodes else -1

            for node_id in range(len(left_children)):
                if node_id == last_cat_node:
                    beg = cat_segments[cat_cnt]
                    size = cat_sizes[cat_cnt]
                    end = beg + size
                    node_cats = cats[beg:end]
                    node_categories.append(node_cats)
                    cat_cnt += 1
                    if cat_cnt == len(cat_nodes):
                        last_cat_node = -1
                    else:
                        last_cat_node = cat_nodes[cat_cnt]
                else:
                    node_categories.append([])

            base_weights = [float(v) for v in tree["base_weights"]]
            loss_changes = [float(v) for v in tree["loss_changes"]]
            sum_hessian = [float(v) for v in tree["sum_hessian"]]

            nodes: List[Node] = []
            for node_id in range(len(left_children)):
                nodes.append(
                    Node(
                        left=left_children[node_id],
                        right=right_children[node_id],
                        parent=parents[node_id],
                        split_idx=split_indices[node_id],
                        split_cond=split_conditions[node_id],
                        default_left=(default_left[node_id] == 1),
                        split_type=SplitType(split_types[node_id]),
                        categories=node_categories[node_id],
                        base_weight=base_weights[node_id],
                        loss_chg=loss_changes[node_id],
                        sum_hess=sum_hessian[node_id],
                    )
                )

            trees.append(Tree(tree_id, nodes))

        self.trees = trees

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        C = self.num_output_group

        logits = np.tile(self.base_score, (N, 1)).astype(float)

        for t_idx, tree in enumerate(self.trees):
            group = self.tree_info[t_idx]  # which class this tree belongs to
            for i in range(N):
                logits[i, group] += tree.predict_one(X[i])

        return logits


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


# ---- load model + labels once ----

with open("xgb_model.json", "r") as f:
    _MODEL_JSON = json.load(f)

_PARSED_MODEL = ParsedModel(_MODEL_JSON)
_CLASS_NAMES = np.load("label_classes.npy", allow_pickle=True)


def predict_all(csv_filename: str):
    dr = DataReader(csv_filename)
    X, Y = dr.to_numpy()  # ignore labels if present
    X = np.asarray(X, dtype=float)

    logits = _PARSED_MODEL.predict_logits(X)
    probs = _softmax(logits)
    pred_idx = np.argmax(probs, axis=1)

    return [_CLASS_NAMES[int(i)] for i in pred_idx], Y


if __name__ == "__main__":
    preds, targets = predict_all("training_data_clean.csv")
    print(len(preds), "preds; first 5:", preds[:10])
    print(len(targets), "Targets; first 5:", targets[:10])
    same = 0
    total = len(targets)
    for pred, target in zip(preds, targets):
        if pred == target:
            same += 1
    print(same/total)