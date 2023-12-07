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
dmd = DMD(svd_rank=-1)
dmd.fit(data.T)  # Transpose data to have time snapshots in columns

# Reconstruct the data using DMD modes
modes = dmd.modes
reconstructed_data = dmd.reconstructed_data.T

error = calculate_error(data, reconstructed_data) +1
print("\n\nerror: ", error)

# Plot the original and reconstructed data
plt.figure(figsize=(12, 6))

for i in range(data.shape[0]):
    plt.plot(data[i], label=f"original Data (Variable {i+1})")


plt.title("Original")
plt.legend()
plt.show()


# ========

plt.figure(figsize=(12, 6))

for i in range(reconstructed_data.shape[0]):
    plt.plot(
        reconstructed_data[i].real,
        label=f"Reduced Data (Variable {i+1})",
        linestyle="--",
    )

plt.title("Original vs Reconstructed Data")
plt.legend()
plt.show()

# ========

plt.figure(figsize=(12, 6))

for i in range(reconstructed_data.shape[0]):
    plt.plot(
        reconstructed_data[i].real,
        label=f"Reduced Data (Variable {i+1})",
        linestyle="--",
    )

for i in range(data.shape[0]):
    plt.plot(data[i], label=f"original Data (Variable {i+1})", linestyle=":")


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


# Determine the number of rows and columns for subplots
num_modes = modes.shape[1]
num_rows = min(2, num_modes)
num_cols = (num_modes + num_rows - 1) // num_rows

# Create subplots for each mode
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))
fig.suptitle("DMD Modes")

# Plot each DMD mode using different types of plots
for i in range(num_modes):
    ax = axs[i // num_cols, i % num_cols]

    # Line plot
    ax.plot(modes[:, i], label=f"Mode {i + 1}", linewidth=2)

    # Scatter plot
    ax.scatter(range(len(modes[:, i])), modes[:, i], label=f"Mode {i + 1}", color="red")

    # Bar plot
    ax.bar(
        range(len(modes[:, i])),
        modes[:, i],
        label=f"Mode {i + 1}",
        color="green",
        alpha=0.5,
    )

    # Stem plot
    ax.stem(modes[:, i], label=f"Mode {i + 1}", basefmt="b", linefmt="g")

    ax.set_title(f"Mode {i + 1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for title
plt.show()
