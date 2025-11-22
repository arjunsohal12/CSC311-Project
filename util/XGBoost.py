import numpy as np
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

from xgboost import XGBClassifier
from util.dataset_reader import DataReader


# -----------------------------
# 0) Constants
# -----------------------------
RESPONSES_PER_USER = 3
HOLDOUT_SEED = random.randint(1, 1000)
SEARCH_SEED = 42


# -----------------------------
# 1) Load + encode data
# -----------------------------
dr = DataReader("training_data_clean.csv", back_compat=False)
X, y = dr.to_numpy()

print("X shape:", X.shape)
print("y shape:", y.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

unique, counts = np.unique(y_encoded, return_counts=True)
print("Class distribution (encoded_label, count):")
print(np.vstack([unique, counts]).T)


# -----------------------------
# 2) User-level train/val/test split
# -----------------------------
def split_data_by_user(
    X, y,
    train_ratio=0.8,
    val_ratio=0.1,
    responses_per_user=3,
    seed=1234
):
    """
    Split by user assuming each user has `responses_per_user`
    consecutive rows. Ensures all responses from a user stay in
    the same split.
    """
    n = len(X)
    n_users = n // responses_per_user
    if n % responses_per_user != 0:
        print(f"Warning: {n} rows not divisible by {responses_per_user}. "
              f"Truncating to {n_users * responses_per_user} rows.")
        n_users = n // responses_per_user

    users = np.arange(n_users)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n_train = int(train_ratio * n_users)
    n_val = int(val_ratio * n_users)

    train_users = users[:n_train]
    val_users = users[n_train:n_train + n_val]
    test_users = users[n_train + n_val:]

    def expand(user_ids):
        idx = []
        for u in user_ids:
            start = u * responses_per_user
            idx.extend(range(start, start + responses_per_user))
        return np.array(idx)

    train_idx = expand(train_users)
    val_idx   = expand(val_users)
    test_idx  = expand(test_users)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"Users: total={n_users}, train={len(train_users)}, "
          f"val={len(val_users)}, test={len(test_users)}")
    print(f"Samples: train={len(train_idx)}, val={len(val_idx)}, "
          f"test={len(test_idx)}")

    return X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx


X_train, y_train, X_val, y_val, X_test, y_test, train_idx, val_idx, test_idx = split_data_by_user(
    X, y_encoded, responses_per_user=RESPONSES_PER_USER, seed=HOLDOUT_SEED
)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# -----------------------------
# 3) Build user groups (for GroupKFold)
# -----------------------------
n_users_total = len(X) // RESPONSES_PER_USER
groups_all = np.repeat(np.arange(n_users_total), RESPONSES_PER_USER)

# Tune ONLY on train+val (keep test untouched)
tune_idx = np.concatenate([train_idx, val_idx])
X_tune, y_tune = X[tune_idx], y_encoded[tune_idx]
groups_tune = groups_all[tune_idx]


# -----------------------------
# 4) Focused param search with user-level CV
# -----------------------------
base = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=SEARCH_SEED,
    tree_method="hist",
)

param_focus = {
    "n_estimators": [300, 500, 700, 900, 1200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "min_child_weight": [3, 5, 8, 10, 15],
    "gamma": [0.0, 0.05, 0.1, 0.2],
    "reg_lambda": [5, 8, 10, 15, 20],
    "reg_alpha": [0.0, 0.5, 1.0, 2.0, 3.0],
    "subsample": [0.5, 0.6, 0.7, 0.8],
    "colsample_bytree": [0.2, 0.3, 0.4, 0.5],
    "tree_method": ["hist"],
}

gkf = GroupKFold(n_splits=5)

search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_focus,
    n_iter=200,
    cv=gkf.split(X_tune, y_tune, groups_tune),
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    random_state=SEARCH_SEED
)

search.fit(X_tune, y_tune)

best_params = search.best_params_
best_cv = search.best_score_

print("\nUser-level RandomizedSearchCV results:")
print("Best params:", best_params)
print("Best user-CV acc:", best_cv)


# -----------------------------
# 5) Train best model w/ class weights + early stopping
# -----------------------------
def make_class_weights(y_labels):
    counts = np.bincount(y_labels)
    w = 1.0 / counts
    return w[y_labels]

train_weights = make_class_weights(y_train)

best_model = XGBClassifier(
    **best_params,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=SEARCH_SEED,
)

best_model.fit(
    X_train, y_train,
    sample_weight=train_weights,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=30,
    verbose=False
)

print("\nEarly-stopped best_iteration:", best_model.best_iteration)


# Evaluate on user split
y_train_pred = best_model.predict(X_train)
y_val_pred   = best_model.predict(X_val)

print("\nBest model performance (user split, early-stopped):")
print("train acc:", accuracy_score(y_train, y_train_pred))
print("val acc:  ", accuracy_score(y_val, y_val_pred))


# -----------------------------
# 6) Refit final model on train+val using best_iteration
# -----------------------------
best_n_estimators = best_model.best_iteration + 1

final_params = dict(best_params)
final_params["n_estimators"] = best_n_estimators  # lock to early-stopped optimum

final_model = XGBClassifier(
    **final_params,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=SEARCH_SEED,
)

trainval_idx = np.concatenate([train_idx, val_idx])
X_trainval, y_trainval = X[trainval_idx], y_encoded[trainval_idx]
trainval_weights = make_class_weights(y_trainval)

final_model.fit(X_trainval, y_trainval, sample_weight=trainval_weights)

y_test_pred = final_model.predict(X_test)

print("\nFinal model performance (trained on train+val, tested on held-out users):")
print("test acc:", accuracy_score(y_test, y_test_pred))


# -----------------------------
# 7) Save final model + label classes
# -----------------------------
final_model.save_model("xgb_model.json")
np.save("label_classes.npy", le.classes_)

# Reload sanity check
reloaded = XGBClassifier()
reloaded.load_model("xgb_model.json")
y_pred_reload = reloaded.predict(X)

print("\nReloaded training accuracy on full data:",
      accuracy_score(y_encoded, y_pred_reload))
