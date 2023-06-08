import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib  # for loading the model

# Load your data from a Feather file
data_file_path = '/data6/tziolkowski/data/raw_txt/running_example/combinedFeatherOutlier'
data_df = pd.read_feather(data_file_path)

# Specify the columns to be used as features
X = data_df[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]

# Check for NaN values in each column
nan_columns = X.columns[X.isna().any()]
num_nan_columns = len(nan_columns)

# Print the number of columns with NaN values
print("Number of columns with NaN values: {}".format(num_nan_columns))

# Replace NaN values with 0
X.fillna(0, inplace=True)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load the trained model from file
model_file_path = '/home/tziolkowski/code/oqf/final/OTD/random_forest_model.pkl'
clf = joblib.load(model_file_path)

# Predict labels for your data
y_pred = clf.predict(X_scaled)

# Add predicted labels to your data DataFrame
data_df['rf_outlier'] = y_pred

# Save the updated data DataFrame with predictions
#output_file_path = '/home/tziolkowski/code/oqf/final/2_OTD/chunk_16.0415--47.7665_raw_with_stats.feather'
data_df.to_feather(data_file_path)

