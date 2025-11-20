from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from util.dataset_reader import DataReader
from sklearn.metrics import accuracy_score
import numpy as np

dr = DataReader("training_data_clean.csv")
X, y = dr.to_numpy()   # y: ['ChatGPT', 'Claude', 'Gemini', ...]
print(X.shape)
print(y.shape)
le = LabelEncoder()
y_encoded = le.fit_transform(y) 

# First split: Train vs (Val+Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_encoded,
    test_size=0.2,          # 40% goes to val+test
    stratify=y_encoded
)

# Second split: Validation vs Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,          # 50% of the remaining 40% → 20% of total
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

model = XGBClassifier(
    # best params from RandomizedSearchCV
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

    # task-specific stuff (same as before)
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)
y_val_pred  = model.predict(X_val)

print("train acc:", accuracy_score(y_train, y_train_pred))
print("test  acc:", accuracy_score(y_test, y_test_pred))
print("val  acc:", accuracy_score(y_val, y_val_pred))

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,       
    X,           # (734, 2693)
    y_encoded,
    cv=skf,
    scoring="accuracy",
    n_jobs=-1
)

print("CV scores:", cv_scores)
print("Mean CV acc:", cv_scores.mean())
print("Std:", cv_scores.std())

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

base = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

param_dist = {
    # --- model size / capacity ---
    "n_estimators": [
        100, 150, 200, 250, 300, 400, 500, 700, 900
    ],
    "max_depth": [
        2, 3, 4, 5, 6, 7, 8
    ],
    "learning_rate": [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15
    ],

    # --- regularization / tree complexity ---
    "min_child_weight": [
        1, 2, 3, 5, 8, 10, 15, 20
    ],
    "gamma": [
        0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0
    ],
    "reg_lambda": [
        0.1, 1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0
    ],
    "reg_alpha": [
        0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0
    ],

    # --- feature / row subsampling ---
    "subsample": [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ],
    "colsample_bytree": [
        0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    ],

    # --- tree method (keep it fast / safe) ---
    "tree_method": [
        "hist"
    ],
}

y_pred = model.predict(X)
print("XGBoost accuracy on full data:", accuracy_score(y_encoded, y_pred))
model.save_model("xgb_model.json")
np.save("label_classes.npy", le.classes_)

base = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# search = RandomizedSearchCV(
#     estimator=base,
#     param_distributions=param_dist,
#     n_iter=300,          # go big if you don't care about runtime
#     cv=skf,
#     scoring="accuracy",
#     n_jobs=-1,
#     verbose=2,
#     random_state=42,
# )

# search.fit(X, y_encoded)

# print("Best params:", search.best_params_)
# print("Best CV acc:", search.best_score_)

from xgboost import XGBClassifier

# Reload the model we just saved and check accuracy again in the SAME script
reloaded = XGBClassifier()
reloaded.load_model("xgb_model.json")

y_pred_reload = reloaded.predict(X)
print("Reloaded accuracy on full data (same script):",
      accuracy_score(y_encoded, y_pred_reload))
