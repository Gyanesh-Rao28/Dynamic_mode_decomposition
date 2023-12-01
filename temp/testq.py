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
    U, Sigma, Vt = svd(X1, full_matrices=False)

    # Truncate matrices
    U_r = U[:, :r]
    Sigma_r = np.diag(Sigma[:r])
    Vt_r = Vt[:r, :]

    # Build A tilde and diagonalize
    A_tilde = U_r.T @ X2 @ Vt_r @ np.linalg.inv(Sigma_r)

    eigenvalues, modes = np.linalg.eig(A_tilde)

    # Compute dynamic modes
    Phi = X2 @ Vt_r @ np.linalg.inv(Sigma_r) @ modes

    # Compute dynamic frequencies
    omega = np.log(np.abs(eigenvalues)) / dt

    x_p = U_r.T @ X1

    # x_p = U_r @ Sigma_r @ Vt_rcls
    

    print(X1.shape)
    print(x_p.shape)

    return Phi, omega, x_p


def get_reduced_matrix(Phi, omega, dt, num_modes):
    # Project modes onto data
    X_hat = Phi * np.diag(np.exp(omega * dt))

    # Truncate to the first 'num_modes' columns
    X_hat = X_hat[:, :num_modes]

    return X_hat


def calculate_error(A, B):
    return np.linalg.norm(A - B, "fro")


# main
file_path = "F:/UROP/dmd/brain_Tumor_dataset.csv"

data = read_csv(file_path)

m, n = data.shape

r = min(m, n)
dt = 1.0

Phi, omega, x_p = dynamic_mode_decomposition(data, dt, r)


# X_hat = get_reduced_matrix(Phi, omega, dt, r)

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
        time_vector[: x_p.shape[1]],
        x_p[i, :],
        label=f"Reduced Data (Variable {i+1})",
        linewidth=2,
    )

plt.title("Original Data V/S Reconstructed Reduced Data")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


error = calculate_error(data[:, 1:], x_p)

error = round(error, 2)

print(error)


# Plot the error over time
plt.figure(figsize=(12, 4))
plt.plot(
    time_vector[1:],
    [error] * (len(time_vector) - 1),
    label="Error",
    color="red",
    linestyle="--",
)
plt.title("Error between Original Data and Reduced Data")
plt.xlabel("Time")
plt.ylabel("Error (Frobenius Norm)")
plt.legend()
plt.show()
