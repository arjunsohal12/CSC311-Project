from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from util.dataset_reader_vis import DataReader
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dr = DataReader("training_data_clean.csv")
X, y = dr.to_numpy()
feature_names = dr.feature_names
le = LabelEncoder()
y_encoded = le.fit_transform(y) 
unique, counts = np.unique(y, return_counts=True)

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
    max_depth=5,
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

# ---- XGBoost global feature importance ----
booster = model.get_booster()
score = booster.get_score(importance_type="gain")  # or "weight", "cover"

importances = np.zeros(len(feature_names))
for k, v in score.items():
    idx = int(k[1:])   # "f123" -> 123
    importances[idx] = v

df_imp = pd.DataFrame({
    "feature": feature_names,
    "gain": importances
})

# keep only word-features
word_df = df_imp[df_imp["feature"].str.startswith("text:")]

top_k = 30
top_words = word_df.sort_values("gain", ascending=False).head(top_k)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# top_words has columns: "feature", "gain" (or "mean_abs_shap")
# feature looks like "text:<col>:<word>"

def parse_text_feature(feat: str):
    # "text:<col>:<word>" -> (col, word)
    parts = feat.split(":")
    col = parts[1]
    word = parts[-1]
    return col, word

top_words_plot = top_words.copy()

# extract question (col) + word
parsed = top_words_plot["feature"].apply(parse_text_feature)
top_words_plot["question"] = parsed.apply(lambda x: x[0])
top_words_plot["word"] = parsed.apply(lambda x: x[1])

# assign a distinct color per question/column
questions = top_words_plot["question"].unique().tolist()
cmap = plt.get_cmap("tab10")  # nice distinct palette
q2color = {q: cmap(i % 10) for i, q in enumerate(questions)}

top_words_plot["color"] = top_words_plot["question"].map(q2color)

# sort already done in top_words, but just to be safe:
top_words_plot = top_words_plot.sort_values(
    by=top_words_plot.columns[-2] if "gain" not in top_words_plot else "gain",
    ascending=False
)

# plot
values_col = "gain" if "gain" in top_words_plot.columns else "mean_abs_shap"

plt.figure(figsize=(11, 8))
plt.barh(
    top_words_plot["word"][::-1],
    top_words_plot[values_col][::-1],
    color=top_words_plot["color"][::-1]
)

plt.xlabel("XGBoost gain importance" if values_col=="gain" else "Mean |SHAP value|")
plt.title(f"Top {len(top_words_plot)} words, colored by question")

# add legend mapping colors->questions
legend_handles = [mpatches.Patch(color=q2color[q], label=q) for q in questions]
plt.legend(handles=legend_handles, title="Text question", loc="lower right")

plt.tight_layout()
plt.gcf().subplots_adjust(left=0.25)  # extra room for words
plt.show()
