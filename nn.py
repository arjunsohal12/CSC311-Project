from util.dataset_reader import *

dr = DataReader("training_data_clean.csv", back_compat=False)
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

    for col in text_cols:
        X.append(np.stack(x_raw[col].to_numpy(), axis = 0)) # (N, d) for each

    # for col in text_cols:
    return np.concatenate(X, axis = -1)

X = getFeatures(X_raw)

print("x shape", X.shape)


# split sets
def split_data(X, y, train_ratio=0.7, val_ratio=0.2, responses_per_user=3, seed=1234):
    """
    Split dataset by user, assuming each user has a fixed number of consecutive responses.

    Ensures that if a user's data is in one split (train/val/test),
    *all* their responses go to that split.
    """
    n = len(X)
    n_users = n // responses_per_user

    if n % responses_per_user != 0:
        print(f"Warning: total rows ({n}) not divisible by responses_per_user ({responses_per_user}).")
        n_users = n // responses_per_user

    users = np.arange(n_users)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)

    n_train_users = int(train_ratio * n_users)
    n_val_users = int(val_ratio * n_users)

    train_users = users[:n_train_users]
    val_users = users[n_train_users:n_train_users + n_val_users]
    test_users = users[n_train_users + n_val_users:]

    def expand_users(user_idx):
        indices = []
        for u in user_idx:
            start = u * responses_per_user
            end = start + responses_per_user
            indices.extend(range(start, end))
        return np.array(indices)

    train_idx = expand_users(train_users)
    val_idx = expand_users(val_users)
    test_idx = expand_users(test_users)

    # Perform splits
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Users: total={n_users}, train={len(train_users)}, val={len(val_users)}, test={len(test_users)}")
    print(f"Samples: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

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
    def __init__(self, features, hidden = [64, 32], classes = 3, dropout = 0.0):
        """
        two layer mlp model
        """

        self.num_features = features
        self.hidden = hidden
        self.num_classes = classes

        self.dropout = dropout
        self.dropout_seed = 0

        self.weights = []
        self.biases = []

        prev = features
        for h in hidden + [classes]:
            self.weights.append(np.random.randn(h, prev) * np.sqrt(2. / prev))
            self.biases.append(np.zeros([h]))
            prev = h

    def activation(self, z):
        return np.maximum(0, z) # relu

    def forward(self, X, training=True):
        """forward pass"""
        a = X
        intermediates = {"A": {0: a}}
        intermediates["Z"] = {}
        intermediates["dropout_mask"] = {}

        for i in range(len(self.hidden)):
            Z = a @ self.weights[i].T + self.biases[i]  # (N, hidden_i)
            A = self.activation(Z)  # relu

            # dropout
            if training and self.dropout > 0.0:
                # np.random.seed(2134+self.dropout_seed)
                self.dropout_seed += 1
                mask = (np.random.rand(*A.shape) > self.dropout).astype(float)
                A *= mask
                A /= (1.0 - self.dropout)
                intermediates["dropout_mask"][i + 1] = mask
            else:
                intermediates["dropout_mask"][i + 1] = np.ones_like(A)

            intermediates["Z"][i+1] = Z
            intermediates["A"][i+1] = A
            a = A

        # Output layer
        Z_out = a @ self.weights[-1].T + self.biases[-1]  # (N, num_classes)
        Y_hat = softmax(Z_out)

        intermediates["Z"][len(self.hidden)+1] = Z_out
        intermediates["A"][len(self.hidden)+1] = Y_hat
        return Y_hat, intermediates

    def backward(self, ts, intermediates):
        """
        back propogation
        """
        L = len(self.hidden) + 1
        N = len(ts)

        g_W = []
        g_B = []

        Y_hat = intermediates["A"][L]
        ts_onehot = np.zeros(Y_hat.shape)
        ts_onehot[np.arange(N), ts] = 1

        d_Z = (Y_hat - ts_onehot)/N

        # back prop
        A_prev = intermediates["A"][L-1]
        g_W.append(d_Z.T @ A_prev)
        g_B.append(np.sum(d_Z, axis=0))
        d_A = d_Z @ self.weights[L-1]

        for i in range(len(self.hidden)-1, -1, -1):
            Z = intermediates["Z"][i+1]
            
            # dropout
            mask = intermediates["dropout_mask"][i + 1]
            d_A *= mask  # apply dropout mask
            d_A /= (1.0 - self.dropout)  # scale as during forward

            A_prev = intermediates["A"][i]

            d_Z = d_A * (Z > 0) # relu der

            g_W.append(d_Z.T @ A_prev)
            g_B.append(np.sum(d_Z, axis=0))

            d_A = d_Z @ self.weights[i]

        g_W.reverse()
        g_B.reverse()

        return g_W, g_B

    def loss(self, ts, y_hat):
        ts_onehot = np.zeros(y_hat.shape)
        ts_onehot[np.arange(y_hat.shape[0]), ts] = 1

        y_hat = np.clip(y_hat, 1e-12, 1 - 1e-12) # numerical stability

        return np.sum(-ts_onehot * np.log(y_hat)) / ts.shape[0]

    def update(self, g_W, g_B, lr=0.01):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * g_W[i]
            self.biases[i] -= lr * g_B[i]


model = NNModel(features=X.shape[1], hidden=[64, 32, 16], classes=3, dropout=0.35)

# Training parameters
epochs = 150
batch_size = 32
learning_rate = 0.002


# Storage for tracking loss
train_losses = []
val_d = {"losses": [], "epochs": []}
print(f"start training for {epochs} epochs\nTraining: {len(X_train)} Val: {len(X_val)}")

def create_mini_batches(X, y, batch_size, epoch, base_seed=67):
    # np.random.seed(base_seed + epoch*epoch)
    n = len(X)
    indices = np.random.permutation(n)
    
    batches = []
    for i in range(0, n, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        batches.append((batch_X, batch_y))
    
    return batches

for epoch in range(epochs):
    batches = create_mini_batches(X_train, y_train, batch_size, epoch)
    epoch_train_losses = []
    epoch_train_acc = []

    for batch_X, batch_y in batches:

        predictions, intermediates = model.forward(batch_X)
        train_loss = model.loss(batch_y, predictions)
        
        # back prop
        g_W, g_B = model.backward(batch_y, intermediates)
        
        # descend that gradient
        model.update(g_W, g_B, lr=learning_rate if epoch > 50 else learning_rate*3)

        epoch_train_losses.append(train_loss)

        train_pred = np.argmax(predictions, axis=1)
        epoch_train_acc.append(np.mean(train_pred == batch_y))
   
    train_loss = np.mean(epoch_train_losses)
    train_acc = np.mean(epoch_train_acc)


    if epoch % 10 == 0 or epoch == epochs - 1:
    # if True:
        Y_hat_val, _ = model.forward(X_val, training=False)
        val_loss = model.loss(y_val, Y_hat_val)
        
        val_pred = np.argmax(Y_hat_val, axis=1)
        val_acc = np.mean(val_pred == y_val)
        
        print(f"Epoch {epoch+1} \nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
              f"Train Acc: {train_acc:.4f}  | Val Acc: {val_acc:.4f}")
        
        val_d["losses"].append(val_loss)
        val_d["epochs"].append(epoch + 1)
    
    train_losses.append(train_loss)

# Final evaluation on test set
Y_hat_test, _ = model.forward(X_test, training=False)
test_loss = model.loss(y_test, Y_hat_test)
test_pred = np.argmax(Y_hat_test, axis=1)
test_acc = np.mean(test_pred == y_test)

print(f"test loss: {test_loss:.4f} | test acc: {test_acc:.4f}")




# as per claude
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_d):
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    
    # Plot validation loss
    # Ensure x-axis aligns if you didn't collect val loss every epoch
    # Since your code uses "if True:", lengths should match
    plt.plot(val_d["epochs"], val_d["losses"], label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    
    plt.title('Model Loss Over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Optional: Set y-axis limit if loss starts extremely high
    # plt.ylim(0, 2.0) 
    
    plt.tight_layout()
    plt.show()

# Call the function
plot_training_history(train_losses, val_d)