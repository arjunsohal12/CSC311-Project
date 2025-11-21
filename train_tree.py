# train_xgb.py
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from xgboost import XGBClassifier


# ---------------------------
# Helpers (match pred.py)
# ---------------------------

def extract_rating(x) -> int:
    """
    Extract leading integer rating from strings like:
    "4 - Very likely", "3 Somewhat likely", or already-int cells.
    Falls back to 0 if missing/unparseable.
    """
    if pd.isna(x):
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    m = re.match(r"^\s*(\d+)", s)
    return int(m.group(1)) if m else 0


def process_multiselect(series: pd.Series, target_tasks: Sequence[str]) -> List[List[str]]:
    """
    Convert comma-separated multi-select column into list-of-lists.
    Keeps only tasks that are exactly in target_tasks (after strip).
    """
    out: List[List[str]] = []
    target_set = set(target_tasks)

    for cell in series.fillna(""):
        tokens = [t.strip() for t in str(cell).split(",") if t.strip()]
        filtered = [t for t in tokens if t in target_set]
        out.append(filtered)

    return out


# ---------------------------
# Config
# ---------------------------

TARGET_TASKS = [
    'Math computations',
    'Writing or debugging code',
    'Data processing or analysis',
    'Explaining complex concepts simply',
    'Converting content between formats (e.g., LaTeX)',
    'Writing or editing essays/reports',
    'Drafting professional text (e.g., emails, résumés)',
    'Brainstorming or generating creative ideas'
]

BEST_COL = 'Which types of tasks do you feel this model handles best? (Select all that apply.)'
SUBOPT_COL = 'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'

ACADEMIC_COL = 'How likely are you to use this model for academic tasks?'
SUBOPT_RATE_COL = 'Based on your experience, how often has this model given you a response that felt suboptimal?'

LABEL_COL = 'label'

OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_features(df: pd.DataFrame,
                   mlb_best: MultiLabelBinarizer,
                   mlb_subopt: MultiLabelBinarizer,
                   target_tasks: Sequence[str],
                   fit: bool = False) -> np.ndarray:
    """
    Build X with exact ordering:
    [academic_numeric, subopt_numeric, best_tasks_encoded, subopt_tasks_encoded]
    If fit=True, fits MLBS; else uses transform.
    """
    best_lists = process_multiselect(df[BEST_COL], target_tasks)
    subopt_lists = process_multiselect(df[SUBOPT_COL], target_tasks)

    if fit:
        best_enc = mlb_best.fit_transform(best_lists)
        subopt_enc = mlb_subopt.fit_transform(subopt_lists)
    else:
        best_enc = mlb_best.transform(best_lists)
        subopt_enc = mlb_subopt.transform(subopt_lists)

    academic_numeric = df[ACADEMIC_COL].apply(extract_rating).values.reshape(-1, 1)
    subopt_numeric = df[SUBOPT_RATE_COL].apply(extract_rating).values.reshape(-1, 1)

    X = np.hstack([academic_numeric, subopt_numeric, best_enc, subopt_enc])
    return X.astype(float)


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[LABEL_COL]).copy()

    mlb_best = MultiLabelBinarizer(classes=TARGET_TASKS)
    mlb_subopt = MultiLabelBinarizer(classes=TARGET_TASKS)

    X = build_features(df, mlb_best, mlb_subopt, TARGET_TASKS, fit=True)

    # labels -> int for XGB
    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].values)

    # same split style you used (with stratify on first split)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tv, y_tv, test_size=0.22, random_state=42, stratify=y_tv
    )

    print("X:", X.shape)
    print("train:", X_train.shape, "valid:", X_valid.shape, "test:", X_test.shape)

    # ---- your tuned params go here ----
    model = XGBClassifier(
        tree_method="hist",
        subsample=0.6,
        reg_lambda=15.0,
        reg_alpha=1.0,
        n_estimators=700,
        min_child_weight=8,
        max_depth=2,
        learning_rate=0.02,
        gamma=0.1,
        colsample_bytree=0.3,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=True
        # you can add early_stopping_rounds=25 if you want
    )

    # quick val/test acc
    val_acc = (model.predict(X_valid) == y_valid).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"val acc: {val_acc:.4f}")
    print(f"test acc: {test_acc:.4f}")

    # ---------------------------
    # Save everything
    # ---------------------------
    model.save_model(str(OUT_DIR / "xgb_model.json"))
    np.save(OUT_DIR / "label_classes.npy", le.classes_, allow_pickle=True)

    with open(OUT_DIR / "mlb_best.pkl", "wb") as f:
        pickle.dump(mlb_best, f)

    with open(OUT_DIR / "mlb_subopt.pkl", "wb") as f:
        pickle.dump(mlb_subopt, f)

    with open(OUT_DIR / "target_tasks.json", "w") as f:
        json.dump(list(TARGET_TASKS), f)

    print("Saved: xgb_model.json, label_classes.npy, mlb_best.pkl, mlb_subopt.pkl, target_tasks.json")


if __name__ == "__main__":
    main("training_data_clean.csv")
