import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.linalg import svd


def read_csv(file_path):
    data = []
    with open(file_path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.append([float(entry) for entry in row])
    return np.array(data)


def dynamic_mode_decomposition(X, dt, r):
    # X: input matrix, each column is a variable, each row is a time snapshot
    # dt: time step

    # Construct the snapshot matrices
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    # Singular Value Decomposition
    U, S, V = svd(X1, full_matrices=False)

    # Truncate matrices
    U = U[:, :r]
    S = np.diag(S[:r])
    V = V[:r, :]
    print(V)

    print(X2.shape)
    print(V.shape)
    # Build A tilde and diagonalize
    A_tilde = U.T @ X2 @ V @ np.linalg.pinv(S)

    W, eigs = np.linalg.eig(A_tilde)

    # Compute dynamic modes
    Phi = X2 @ V @ np.linalg.inv(S) @ W

    return Phi, A_tilde


# main
file_path = "F:/UROP/dmd/indexData.csv"

data = read_csv(file_path)

data = data.T

m, n = data.shape

r = min(m, n)

dt = 1.0

phi, A_tilde = dynamic_mode_decomposition(data, dt, r)

# Generate time vector
time_vector = np.arange(0, dt * data.shape[1], dt)

# Number of variables in your data
num_variables = data.shape[0]

# Plot the original, reconstructed_reduced_data for each variable
plt.figure(figsize=(12, 8))

for i in range(3):
    plt.plot(
        time_vector[: data.shape[1]],
        data[i, :],
        label=f"original Data (Variable {i+1})",
        linewidth=2,
        linestyle=":",
    )
    plt.plot(
        time_vector[: A_tilde.shape[1]],
        A_tilde[i - 1, :],
        label=f"Reduced Data (Variable {i+1})",
        linewidth=2,
    )

plt.title("Original Data V/S Reconstructed Reduced Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
