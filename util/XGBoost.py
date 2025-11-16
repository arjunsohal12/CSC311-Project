from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from util.dataset_reader import DataReader

dr = DataReader("training_data_clean.csv")
X, y = dr.to_numpy()   # y: ['ChatGPT', 'Claude', 'Gemini', ...]
print(X.shape)
print(y.shape)
le = LabelEncoder()
y_encoded = le.fit_transform(y) 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

from xgboost import XGBClassifier

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(
    n_estimators=300,       # not huge
    max_depth=4,           # shallow trees
    learning_rate=0.05,

    min_child_weight=10,
    gamma=1.0,
    subsample=0.7,
    colsample_bytree=0.4,
    reg_lambda=15.0,       # strong L2
    reg_alpha=4.0,         # strong L1

    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    n_jobs=-1,
    random_state=42,
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

print("train acc:", accuracy_score(y_train, y_train_pred))
print("test  acc:", accuracy_score(y_test, y_test_pred))

from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,       # your current best XGBClassifier
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
    # --- model size ---
    "n_estimators": [
        100, 150, 200, 250, 300, 350, 400, 500
    ],
    "max_depth": [
        2, 3, 4, 5, 6  
    ],
    "learning_rate": [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1
    ],
    
    # --- regularization ---
    "min_child_weight": [
        1, 2, 3, 5, 8, 10
    ],
    "gamma": [
        0, 0.1, 0.3, 0.5, 1.0, 2.0
    ],
    "reg_lambda": [
        1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0
    ],
    "reg_alpha": [
        0.0, 0.5, 1.0, 2.0, 5.0
    ],

    # --- feature subsampling ---
    "subsample": [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ],
    "colsample_bytree": [
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    ],

    # --- tree method ---
    "tree_method": [
        "hist"   
    ],
}


search = RandomizedSearchCV(
    base,
    param_distributions=param_dist,
    n_iter=40,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
    random_state=42,
)

search.fit(X, y_encoded)

print("Best params:", search.best_params_)
print("Best CV acc:", search.best_score_)
