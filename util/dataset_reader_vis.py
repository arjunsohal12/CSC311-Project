from __future__ import annotations
import pandas as pd
from typing import List, Optional, Union
import numpy as np
import string

TABLE = str.maketrans('', '', string.punctuation)

numeric_cols = [
    'How likely are you to use this model for academic tasks?',
    'How often do you expect this model to provide responses with references or supporting evidence?',
    "How often do you verify this model's responses?",
    'Based on your experience, how often has this model given you a response that felt suboptimal?',

]
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    'Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?',
    "When you verify a response from this model, how do you usually go about it?",
]
hot_cols = [
    'Which types of tasks do you feel this model handles best? (Select all that apply.)',
    'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)',
]
class DataReader:
    def __init__(self, path: str, **read_csv_kwargs):
        self.path = path
        self.read_csv_kwargs = read_csv_kwargs
        self.hot_vocab = {}   # col -> list of categories
        self.text_vocab = {}  # col -> list of words
        self.feature_names = None
        self.load()

    def to_numpy(self):
        blocks = []
        feature_names = []

        # one-hot multi-select columns
        for col in hot_cols:
            mat = np.stack(self.X[col].to_numpy(), axis=0)  # (N, d_hot)
            blocks.append(mat)

            vocab = self.hot_vocab[col]
            feature_names.extend([f"hot:{col}:{v}" for v in vocab])

        # numeric columns
        for col in numeric_cols:
            vec = np.expand_dims(self.X[col].to_numpy(), axis=1)  # (N, 1)
            blocks.append(vec)
            feature_names.append(f"num:{col}")

        # text bag-of-words columns
        for col in text_cols:
            mat = np.stack(self.X[col].to_numpy(), axis=0)  # (N, d_text)
            blocks.append(mat)

            vocab = self.text_vocab[col]
            feature_names.extend([f"text:{col}:{w}" for w in vocab])

        X = np.concatenate(blocks, axis=-1)
        Y = self.labels.to_numpy()

        self.feature_names = feature_names
        return X, Y

    def load(self) -> "DataReader":
        self.X = pd.read_csv(self.path, **self.read_csv_kwargs)
        self.X.dropna(inplace=True)
        self.labels = self.X.pop("label")
        self.X = self.X.drop("student_id", axis=1)

        # numeric parsing
        for col in numeric_cols:
            self.X[col] = self.X[col].str.split(" ").str[0].astype(int)

        # hot cols vocab + vectorize
        for col in hot_cols:
            selections = set()
            for cell in self.X[col].dropna():
                for item in cell.split(","):
                    selections.add(item.strip())
            selections = sorted(selections)
            selections = [x for x in selections if len(x) >= len("Math computations")]

            self.hot_vocab[col] = selections

            def to_vector(cell):
                tokens = {x.strip() for x in str(cell).split(",") if x.strip()}
                return [1 if cat in tokens else 0 for cat in selections]

            self.X[col] = self.X[col].fillna("").apply(to_vector)

        # text cols vocab + vectorize
        for col in text_cols:
            def remove_punctuation(cell):
                return cell.translate(TABLE).lower()

            self.X[col] = self.X[col].fillna("").apply(remove_punctuation)

            selections = set()
            for cell in self.X[col].dropna():
                for item in cell.split(" "):
                    if item.strip():
                        selections.add(item.strip())

            selections = sorted(selections)
            self.text_vocab[col] = selections

            def to_vector(cell):
                tokens = {x.strip() for x in str(cell).split(" ") if x.strip()}
                return [1 if word in tokens else 0 for word in selections]

            self.X[col] = self.X[col].fillna("").apply(to_vector)

        return self
