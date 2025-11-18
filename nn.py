from util.dataset_reader import *

dr = DataReader("training_data_clean.csv")
X_raw = dr.X   # y: ['ChatGPT', 'Claude', 'Gemini', ...]
y = dr.labels.to_numpy()
y_index, y = np.unique(y, return_inverse=True)
print("y shape", y.shape)
# print(y)
# print(y_index)
# extract the features

def getFeatures(x_raw, norm = True):
    X = []
    for col in hot_cols:
        X.append(np.stack(x_raw[col].to_numpy(), axis = 0)) # (N, d) for each

    for col in numeric_cols:
        # normalization 
        numeric_values = x_raw[col].to_numpy().astype(float)
        # z = (x - mean) / std
        mean = np.nanmean(numeric_values)
        std = np.nanstd(numeric_values)
        if std == 0:
            std = 1.0  # avoid division by zero if constant column
        normalized = (numeric_values - mean) / std
        X.append(np.expand_dims(normalized if norm else numeric_values.astype(int), axis = 1)) # (N, d)

    # for col in text_cols:
    return np.concatenate(X, axis = -1)

X = getFeatures(X_raw)

print("x shape", X.shape)


np.random.seed(42)
# split sets
def split_data(X, y, train_ratio=0.7, val_ratio=0.2):
    n = len(X)
    idx = np.random.permutation(n)
    n_train, n_val = int(n*train_ratio), int(n*(train_ratio+val_ratio))
    X_train, X_val, X_test = X[idx[:n_train]], X[idx[n_train:n_val]], X[idx[n_val:]]
    y_train, y_val, y_test = y[idx[:n_train]], y[idx[n_train:n_val]], y[idx[n_val:]]
    return X_train, y_train, X_val, y_val, X_test, y_test

sd = split_data(X, y) # 7:2:1 split

X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

# from lab 6 
# type shit
def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation (see above).

    Parameters:
        `z` - a numpy array of shape (K,) or (N, K)

    Returns: a numpy array with the same shape as `z`, with the softmax
        activation applied to each row of `z`
    """

    if z.ndim == 1: # vector of shape (K,)
        zm = z - np.max(z)
        exp = np.exp(zm)
        return exp / np.sum(exp)
    else:
        zm = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(zm)
        return exp / np.sum(exp, axis=1, keepdims=True)
    
class NNModel():
    def __init__(self, features, hidden = [64, 32], classes = 3):
        """
        two layer mlp model
        """

        self.num_features = features
        self.hidden = hidden
        self.num_classes = classes

        self.weights = []
        self.biases = []

        prev = features
        for h in hidden + [classes]:
            self.weights.append(np.random.randn(h, prev) * 0.01)
            self.biases.append(np.zeros([h]))
            prev = h

        
        def 

        def forward(self, X):
            """forward pass"""
            a = X
            caches = []
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                z = a @ W.T + b
                a = np.maximum(0, z)  # relu
                caches.append((z, a))

            # output layer
            z_out = a @ self.weights[-1].T + self.biases[-1]
            y_hat = softmax(z_out)
            return y_hat, caches

        def backward(self, ts):
            """
            back propogation
            """
            pass

        def loss(self, ts):
            pass

        def update(self, alpha):
            pass