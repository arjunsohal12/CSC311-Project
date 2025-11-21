# pred.py
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


# ---------------------------
# Helpers (must match train)
# ---------------------------

def extract_rating(x) -> int:
    if pd.isna(x):
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    m = re.match(r"^\s*(\d+)", s)
    return int(m.group(1)) if m else 0


def process_multiselect(series: pd.Series, target_tasks: Sequence[str]) -> List[List[str]]:
    out: List[List[str]] = []
    target_set = set(target_tasks)

    for cell in series.fillna(""):
        tokens = [t.strip() for t in str(cell).split(",") if t.strip()]
        filtered = [t for t in tokens if t in target_set]
        out.append(filtered)

    return out


# ---------------------------
# Columns
# ---------------------------

BEST_COL = 'Which types of tasks do you feel this model handles best? (Select all that apply.)'
SUBOPT_COL = 'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'

ACADEMIC_COL = 'How likely are you to use this model for academic tasks?'
SUBOPT_RATE_COL = 'Based on your experience, how often has this model given you a response that felt suboptimal?'


# ---------------------------
# ParsedModel (unchanged)
# ---------------------------

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
        self.base_score: List[float] = json.loads(self.learner_model_shape["base_score"])

        gbm = learner["gradient_booster"]["model"]
        self.tree_info = gbm["tree_info"]

        model_shape = gbm["gbtree_model_param"]
        self.num_trees = int(model_shape["num_trees"])

        j_trees = gbm["trees"]
        trees: List[Tree] = []

        for i in range(self.num_trees):
            tree = j_trees[i]
            tree_id = int(tree["id"])
            assert tree_id == i

            left_children = _to_int_list(tree["left_children"])
            right_children = _to_int_list(tree["right_children"])
            parents = _to_int_list(tree["parents"])
            split_conditions = [float(v) for v in tree["split_conditions"]]
            split_indices = _to_int_list(tree["split_indices"])
            default_left = _to_int_list(tree["default_left"])
            split_types = _to_int_list(tree["split_type"])

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
            group = self.tree_info[t_idx]
            for i in range(N):
                logits[i, group] += tree.predict_one(X[i])

        return logits


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


# ---------------------------
# Load saved model + encoders
# ---------------------------

with open("xgb_model.json", "r") as f:
    _MODEL_JSON = json.load(f)
_PARSED_MODEL = ParsedModel(_MODEL_JSON)

_CLASS_NAMES = np.load("label_classes.npy", allow_pickle=True)

with open("mlb_best.pkl", "rb") as f:
    _MLB_BEST = pickle.load(f)

with open("mlb_subopt.pkl", "rb") as f:
    _MLB_SUBOPT = pickle.load(f)

with open("target_tasks.json", "r") as f:
    _TARGET_TASKS = json.load(f)


def build_features_from_csv(csv_filename: str) -> np.ndarray:
    df = pd.read_csv(csv_filename).copy()

    best_lists = process_multiselect(df[BEST_COL], _TARGET_TASKS)
    subopt_lists = process_multiselect(df[SUBOPT_COL], _TARGET_TASKS)

    best_enc = _MLB_BEST.transform(best_lists)
    subopt_enc = _MLB_SUBOPT.transform(subopt_lists)

    academic_numeric = df[ACADEMIC_COL].apply(extract_rating).values.reshape(-1, 1)
    subopt_numeric = df[SUBOPT_RATE_COL].apply(extract_rating).values.reshape(-1, 1)

    X = np.hstack([academic_numeric, subopt_numeric, best_enc, subopt_enc])
    return X.astype(float)


def predict_all(csv_filename: str):
    df = pd.read_csv(csv_filename)
    Y = df["label"].values if "label" in df.columns else None

    X = build_features_from_csv(csv_filename)

    logits = _PARSED_MODEL.predict_logits(X)
    probs = _softmax(logits)
    pred_idx = np.argmax(probs, axis=1)

    preds = [_CLASS_NAMES[int(i)] for i in pred_idx]
    return preds, Y


if __name__ == "__main__":
    preds, targets = predict_all("training_data_clean.csv")
    print(len(preds), "preds; first 10:", preds[:10])
    if targets is not None:
        same = sum(int(p == t) for p, t in zip(preds, targets))
        print("accuracy:", same / len(targets))
