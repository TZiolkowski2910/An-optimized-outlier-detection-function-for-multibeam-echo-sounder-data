import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

pd.set_option('display.max_columns', None)

# Path to the combined features file
combined_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/combined_predictions.feather'

# Load the combined features DataFrame
df = pd.read_feather(combined_file_path)

# Define the ranges for each column
std_dev_depth_ranges = [0, 100, 200, 300, 400, 500, 600, np.inf]
normalized_distance_ranges = [-20, -14, -11, -8, -5, -2, 0, 2, 5, 8, 11, 23]
mean_depth_ranges = np.arange(-6500, -1499, 500)

outlier_columns = ['db_outlier', 'lof_outlier', 'kmeans_outlier', 'std_outlier', 'mad_outlier', 'iqr_outlier', 'lr_outlier', 'mlp_outlier', 'rf_outlier']
true_outlier_column = 'outlier'

# Create a list to hold the results
results = []

for column in outlier_columns:
    for i in range(len(std_dev_depth_ranges)-1):
        range_name = f"{std_dev_depth_ranges[i]}-{std_dev_depth_ranges[i+1]}"
        range_mask = (df['std_dev_depth_100m'] >= std_dev_depth_ranges[i]) & (df['std_dev_depth_100m'] < std_dev_depth_ranges[i+1])
        y_true = df[true_outlier_column][range_mask]
        y_pred = df[column][range_mask]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append([column, "std_dev_depth_100m", range_name, precision, recall, f1])

    for i in range(len(normalized_distance_ranges)-1):
        range_name = f"{normalized_distance_ranges[i]}-{normalized_distance_ranges[i+1]}"
        range_mask = (df['normalized_distance_100m'] >= normalized_distance_ranges[i]) & (df['normalized_distance_100m'] < normalized_distance_ranges[i+1])
        y_true = df[true_outlier_column][range_mask]
        y_pred = df[column][range_mask]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append([column, "normalized_distance_100m", range_name, precision, recall, f1])

    for i in range(len(mean_depth_ranges)-1):
        range_name = f"{mean_depth_ranges[i]}-{mean_depth_ranges[i+1]}"
        range_mask = (df['mean_depth_100m'] >= mean_depth_ranges[i]) & (df['mean_depth_100m'] < mean_depth_ranges[i+1])
        y_true = df[true_outlier_column][range_mask]
        y_pred = df[column][range_mask]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        results.append([column, "mean_depth_100m", range_name, precision, recall, f1])

# Create a DataFrame from the results list
results_df = pd.DataFrame(results, columns=['column', 'parameter', 'range', 'precision', 'recall', 'f1'])

# Save the DataFrame to a CSV file
results_df.to_csv('outlier_results.csv', index=False)
