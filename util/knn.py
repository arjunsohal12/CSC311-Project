# train_knn.py

import numpy as np
from sklearn.preprocessing import LabelEncoder
from util.dataset_reader import DataReader

# 1. Load data with your DataReader
dr = DataReader("training_data_clean.csv")
X, y = dr.to_numpy()              # X: (N, D), y: strings

X = X.astype(np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# 2. Encode labels as ints
le = LabelEncoder()
y_int = le.fit_transform(y)       # 0, 1, 2 for ['ChatGPT', 'Claude', 'Gemini', ...]

# 3. Standardize features (helps a lot for KNN in high dim)
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1.0               # avoid div by zero

X_norm = (X - mean) / std

print("Normalized X shape:", X_norm.shape)

# 4. Choose k (you can tune this; 7 is a decent starting point)
best_k = 7

# 5. Save everything needed for pred.py
np.save("knn_X_train.npy", X_norm)
np.save("knn_y_train.npy", y_int)
np.save("knn_mean.npy", mean)
np.save("knn_std.npy", std)
np.save("label_classes.npy", le.classes_)  # reuse same file name as before if you like
np.save("knn_k.npy", np.array([best_k], dtype=np.int64))

print("Saved knn_X_train.npy, knn_y_train.npy, knn_mean.npy, knn_std.npy, label_classes.npy, knn_k.npy")
