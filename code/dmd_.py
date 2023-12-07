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

    # Build A tilde and diagonalize
    A_tilde = U.T @ X2 @ V @ np.linalg.pinv(S)

    W, eigs = np.linalg.eig(A_tilde)

    # Compute dynamic modes
    Phi = X2 @ V @ np.linalg.inv(S) @ W

    lambda_diag = np.diag(np.exp(np.log(eigs) / dt))  # dicrete time eigenvalue
    omega = np.log(eigs) / dt  # continous time eigenvalue

    return Phi, A_tilde, omega, lambda_diag


# main
file_path = "F:/UROP/dmd/fluid_motion.csv"

data = read_csv(file_path)
data = data.T

m, n = data.shape
r = min(m, n)
dt = 1.0

phi, A_tilde, omega, lambda_diag = dynamic_mode_decomposition(data, dt, r)

# Choose a specific column for analysis
column_index = 0

x = data[:, column_index]
x = x.reshape(-1, 1)  # Reshape x to make it a 2D array

b = np.linalg.lstsq(phi, x, rcond=None)[0]

# Initialize DMD dynamics matrix
time_dynamics = np.zeros((r, data.shape[1]))

# Compute DMD dynamics
for iter in range(data.shape[1]):
    time_dynamics[:, iter] = b.flatten() * np.exp(omega * iter * dt)

# Compute DMD solution
X_dmd = np.dot(phi, time_dynamics)


# Generate time vector
time_vector = np.arange(0, dt * data.shape[1], dt)

# Plot the original and reconstructed reduced data for the chosen variable
plt.figure(figsize=(12, 8))
plt.plot(
    time_vector,
    data[column_index, :],
    label="Original Data",
    linewidth=2,
    linestyle=":",
)
plt.plot(time_vector, X_dmd[0, :], label="Reconstructed Reduced Data", linewidth=2)
plt.title("Original Data vs. Reconstructed Reduced Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
