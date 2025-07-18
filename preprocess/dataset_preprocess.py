"""
Load the dataset and preprocess it for training.
"""

import csv
import numpy as np


def load_matrix_weigths_labels(file_path, bias_factor=1.0):
    """
    Load dataset from a CSV file and return the feature evaluation matrix,
      sample weights, and sample labels.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        bias_factor (float): Factor to adjust sample weights for positive samples.
    """

    feature_eval_matrix = []
    sample_weights = []
    sample_labels = []

    # Read the CSV file and extract features and labels
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            feature_eval_matrix.append([float(x) for x in row[:-2]])
            sample_labels.append(int(row[-1]))

    # Convert the feature evaluation matrix to a NumPy array
    feature_eval_matrix = np.array(feature_eval_matrix)

    # Substitute the zero values (missing values) with the median value for each feature
    for col in range(feature_eval_matrix.shape[1]):
        col_values = feature_eval_matrix[:, col]
        median_value = np.median(
            col_values[col_values != 0]
        )  # Compute median excluding zeros
        feature_eval_matrix[:, col][
            col_values == 0
        ] = median_value  # Replace zeros with the median

    # Engineer new features based on existing ones
    updated_matrix = feature_eval_matrix.copy()

    # Multiply each column with all other columns to create new features
    for i in range(feature_eval_matrix.shape[1]):
        for j in range(i + 1, feature_eval_matrix.shape[1]):
            new_feature = (
                feature_eval_matrix[:, i] * feature_eval_matrix[:, j]
            ).reshape(-1, 1)
            updated_matrix = np.hstack((updated_matrix, new_feature))

    # Sum pairs of original matrix features to create new features
    for i in range(feature_eval_matrix.shape[1]):
        for j in range(i + 1, feature_eval_matrix.shape[1]):
            new_feature = (
                feature_eval_matrix[:, i] + feature_eval_matrix[:, j]
            ).reshape(-1, 1)
            updated_matrix = np.hstack((updated_matrix, new_feature))

    # Sum pairs of squared features to create new features
    for i in range(feature_eval_matrix.shape[1]):
        for j in range(i + 1, feature_eval_matrix.shape[1]):
            new_feature = (
                feature_eval_matrix[:, i] ** 2 + feature_eval_matrix[:, j] ** 2
            ).reshape(-1, 1)
            updated_matrix = np.hstack((updated_matrix, new_feature))

    # Convert the zero labels to -1 for compatibility with AdaBoost
    sample_labels = [1 if label == 1 else -1 for label in sample_labels]

    # Transpose the feature evaluation matrix to have features as rows and samples as columns
    updated_matrix = np.array(updated_matrix).T

    # Assign sample weights based on the labels
    for label in sample_labels:
        if label == 1:
            sample_weights.append(bias_factor * 1.0)  # Positive class
        else:
            sample_weights.append(0.5)  # Negative class

    # Normalize the sample weights
    sample_weights = np.array(sample_weights)
    sample_weights /= np.sum(sample_weights)

    return (
        np.array(updated_matrix),
        np.array(sample_weights),
        np.array(sample_labels),
    )


if __name__ == "__main__":
    # Example usage
    FILE_PATH = "./dataset/pima_indians_diabetes.csv"  # Replace with your dataset path
    feat_mat, w, l = load_matrix_weigths_labels(FILE_PATH)

    print("Feature Evaluation Matrix:")
    print(feat_mat)
    print("Sample Weights:")
    print(w)
    print("Sample Labels:")
    print(l)
