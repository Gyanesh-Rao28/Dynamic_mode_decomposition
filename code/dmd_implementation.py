import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD


# Function to read data from CSV file
def read_csv(file_path):
    return np.genfromtxt(file_path, delimiter=",")


# Load data from CSV file
file_path = "F:/UROP/dmd/indexData.csv"  # Replace with your actual file path
data = read_csv(file_path)

# Perform DMD
dmd = DMD(svd_rank=-1)  # Set svd_rank to -1 for full-rank decomposition
dmd.fit(data.T)  # Transpose data to have time snapshots in columns

# Reconstruct the data using DMD modes
modes = dmd.modes
reconstructed_data = dmd.reconstructed_data.T

# Plot the original and reconstructed data
plt.figure(figsize=(12, 6))

for i in range(data.shape[0]):
    plt.plot(data[i], 
    label=f"original Data (Variable {i+1})",
    linestyle=":")

for i in range(reconstructed_data.shape[0]):
    plt.plot(
        reconstructed_data[i].real,
        label=f"Reduced Data (Variable {i+1})",
        linestyle="--",
    )

plt.title("Original vs Reconstructed Data")
plt.legend()
plt.show()
