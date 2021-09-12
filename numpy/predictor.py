import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loss(y_pred, y_true):
    error = np.subtract(y_true, y_pred)
    loss_value = np.square(error).mean()
    return loss_value


def loss(w, X, y):
    # Add Constant column with ones at the start
    X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
    y_true = y
    y_pred = np.dot(X, w)
    error = np.subtract(y_true, y_pred)
    loss_value = np.square(error).mean()
    return loss_value


def grad(w_k, X, y):
    # Add Constant column with ones at the start
    X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)

    loss_gradient = - (2.0 / (X.shape[0])) * np.dot(X.transpose(), np.subtract(y, np.dot(X, w_k)))

    return loss_gradient


def grad_descent(w_init, alpha, X, y, maxiter=500, eps=1e-2):
    losses = []
    weights = [w_init]

    count_iteration = 0
    w_k = weights[-1]

    grad_value = grad(w_k, X, y)

    while count_iteration < maxiter and np.max(np.abs(grad_value)) > eps:
        print("count_iteration: " + str(count_iteration))

        count_iteration += 1

        w_k = w_k - alpha * grad_value

        loss_value_k = loss(w_k, X, y)
        grad_value = grad(w_k, X, y)

        weights.append(w_k)
        losses.append(loss_value_k)

    return weights, losses


def train_model(X, y):
    """

    :rtype: TrainedWeights of the model
    """

    # Try multiple values of alpha to expirement and observe difference b/w
    # Convergence rate
    # Divergence at higher values of alpha
    alphas = [0.001, 0.005, 0.01, 0.01]
    initial_weight_list = [np.zeros(shape=(X.shape[1] + 1, 1)), np.ones(shape=(X.shape[1] + 1, 1)),
                           np.random.rand(X.shape[1] + 1, 1)]

    best_regression_weights = None
    best_regression_loss = np.inf

    model_losses = {}
    model_weights = {}

    for alpha_index in range(len(alphas)):
        alpha = alphas[alpha_index]
        for w_index in range(len(initial_weight_list)):
            w_init = initial_weight_list[w_index]
            weights, losses = grad_descent(w_init, alpha, X, y)

            model_losses[str(alpha_index) + "_" + str(w_index)] = losses[-1]
            model_weights[str(alpha_index) + "_" + str(w_index)] = weights[-1]

            if losses[-1] < best_regression_loss:
                best_regression_loss = losses[-1]
                best_regression_weights = weights[-1]

            plt.scatter(np.arange(len(losses)), losses)
            plt.xlabel("Iterations for alpha: " + str(alpha))
            plt.ylabel("Loss Value for w_index: " + str(w_index))
            plt.legend("Alpha = " + str(alpha))
            plt.show()

    return best_regression_weights


def predict(X, w):
    X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
    y_pred = np.dot(X, w)
    return y_pred


if __name__ == '__main__':
    df = pd.read_csv("Housing.csv")
    features = ["area"]
    df_y = df["price"].to_numpy()
    y = np.log(df_y)

    df_x = df[features].to_numpy()

    # Visualize the scatter plot b/w area and price variables to get an intuition about the relationship
    plt.figure(figsize=(8, 8))
    plt.scatter(df_x, df_y)
    plt.show()

    # Reshape the variables to get a matrix with 1 single column vector.
    # This is required for avoiding error in matrix multiplication
    y = np.reshape(y, [y.shape[0], 1])

    X = np.log(df_x)
    X = np.reshape(X, [X.shape[0], 1])

    trained_weights = train_model(X, y)
    predicted_values = predict(X)
    loss_on_test = loss(predicted_values, y)

    print("MSE on test data: " + str(loss_on_test))
