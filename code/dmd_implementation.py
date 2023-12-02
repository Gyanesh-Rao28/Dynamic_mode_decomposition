import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD


# Function to read data from CSV file
def read_csv(file_path):
    return np.genfromtxt(file_path, delimiter=",")


def calculate_error(A, B):
    return np.linalg.norm(A - B, "fro")


# Load data from CSV file
file_path = "F:/UROP/dmd/brain_Tumor_dataset.csv"  # Replace with your actual file path
data = read_csv(file_path)

# Perform DMD
dmd = DMD(svd_rank=-1)  # Set svd_rank to -1 for full-rank decomposition
dmd.fit(data.T)  # Transpose data to have time snapshots in columns

# Reconstruct the data using DMD modes
modes = dmd.modes
reconstructed_data = dmd.reconstructed_data.T

error = calculate_error(data, reconstructed_data)
print("\n\nerror: ", error)

# Plot the original and reconstructed data
plt.figure(figsize=(12, 6))

for i in range(data.shape[0]):
    plt.plot(data[i], label=f"original Data (Variable {i+1})", linestyle=":")

for i in range(reconstructed_data.shape[0]):
    plt.plot(
        reconstructed_data[i].real,
        label=f"Reduced Data (Variable {i+1})",
        linestyle="--",
    )

plt.title("Original vs Reconstructed Data")
plt.legend()
plt.show()


# Plot the error over time

# Generate time vector
dt = 1.0
time_vector = np.arange(0, dt * data.shape[1], dt)

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

# modes
# plt.figure(figsize=(12, 6))

# for i in range(modes.shape[1]):
#     plt.plot(modes[:, i].real, label=f"DMD Mode {i+1}")

# plt.title("DMD Modes")
# plt.legend()
# plt.show()

# Plot the DMD modes as histograms
plt.figure(figsize=(12, 6))

for i in range(modes.shape[1]):
    plt.hist(modes[:, i].real, bins=50, alpha=0.5, label=f"DMD Mode {i+1}")

plt.title("Histogram of DMD Modes")
plt.xlabel("Real Part of Mode")
plt.ylabel("Frequency")
plt.legend()
plt.show()
