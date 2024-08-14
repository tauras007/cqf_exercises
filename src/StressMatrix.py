import numpy as np
import pandas as pd

# Step 1: Generate the original correlation matrix # For reproducibility
data = np.array([[1, 0.4, 0.3, 0.3],
                         [0.4, 1, 0.27, 0.42],
                         [0.3, 0.27, 1, 0.5],
                         [0.3, 0.42, 0.5, 1]])
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
corr_matrix = df.corr()

print("Original Correlation Matrix:")
print(corr_matrix)


# Step 2: Define the stress function
def stress_correlation(matrix, stress_factors, upper_limit=0.99):
    stressed_matrices = {}

    for factor in stress_factors:
        stressed_matrix = matrix.copy()
        for i in range(stressed_matrix.shape[0]):
            for j in range(stressed_matrix.shape[1]):
                if i != j:  # Only apply to off-diagonal elements
                    stressed_value = stressed_matrix.iloc[i, j] * factor
                    stressed_matrix.iloc[i, j] = min(stressed_value, upper_limit)
        # Ensure diagonal elements are 1
        np.fill_diagonal(stressed_matrix.values, 1)
        stressed_matrices[f"×{factor}"] = stressed_matrix

    return stressed_matrices


# Define stress factors
stress_factors = [1, 1.3, 1.8]

# Stress the correlation matrix
stressed_corr_matrices = stress_correlation(corr_matrix, stress_factors)

# Step 3: Display the stressed correlation matrices
for factor, matrix in stressed_corr_matrices.items():
    print(f"\nStressed Correlation Matrix {factor}:")
    print(matrix)
