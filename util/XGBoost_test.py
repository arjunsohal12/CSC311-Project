# util/XGBoost_test.py

import hashlib

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from util.dataset_reader import DataReader


def hash_array(a: np.ndarray) -> str:
    a = np.asarray(a, dtype=np.float32)
    return hashlib.md5(a.tobytes()).hexdigest()


def main():
    # 1. Load data with the SAME DataReader and CSV
    dr = DataReader("training_data_clean.csv")
    X, y = dr.to_numpy()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("HASH X:", hash_array(X))
    print("HASH y:", hashlib.md5(np.asarray(y).tobytes()).hexdigest())

    # 2. Use EXACT SAME label mapping as training
    classes = np.load("label_classes.npy", allow_pickle=True)
    print("classes from disk:", classes)

    le = LabelEncoder()
    le.classes_ = classes
    y_encoded = le.transform(y)

    # 3. Load the classifier exactly like in util.XGBoost
    model = XGBClassifier()
    model.load_model("xgb_model.json")

    # 4. Predict on the full X and compare to y_encoded
    y_pred = model.predict(X)
    acc = accuracy_score(y_encoded, y_pred)
    print("XGBoost accuracy on full data (loaded from json in test):", acc)


if __name__ == "__main__":
    main()
