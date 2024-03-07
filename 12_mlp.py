import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the model

# Load the training data from Feather file
train_file_path = '/data6/tziolkowski/oqf/train3135/chunks/OTD_features_train.feather'
train_df = pd.read_feather(train_file_path)

# Drop NaN values from the training dataset
train_df = train_df.dropna()

# Specify the columns to be used as features and target variable
X = train_df[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y = train_df['outlier']

# Split the training data into training and evaluation sets (80% train, 20% eval)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

# Create an MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(32, 32, 32), activation='relu', solver='sgd', batch_size=32,
                    learning_rate='adaptive', max_iter=40, early_stopping=True, tol=0.002, random_state=42,
                    validation_fraction=0.2, n_iter_no_change=3, verbose=1)

# Fit the classifier to the training part of the data
clf.fit(X_train_scaled, y_train)

# Predict labels for the evaluation part of the data
y_pred_eval = clf.predict(X_eval_scaled)

# Calculate precision, recall, and F1 score using the evaluation part of the data
precision_eval = precision_score(y_eval, y_pred_eval)
recall_eval = recall_score(y_eval, y_pred_eval)
f1_eval = f1_score(y_eval, y_pred_eval)

# Print the results for the evaluation data
print("Evaluation Precision: {:.2f}".format(precision_eval))
print("Evaluation Recall: {:.2f}".format(recall_eval))
print("Evaluation F1 Score: {:.2f}".format(f1_eval))

# Load the separate test data from Feather file
test_file_path = '/data6/tziolkowski/oqf/eval3135/chunks/OTD_features_eval.feather'
test_df = pd.read_feather(test_file_path)

# Drop NaN values in test data and scale
test_df = test_df.dropna()
X_test = test_df[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
X_test_scaled = scaler.transform(X_test)  # Use the scaler fitted on the training data

# Predict labels for the separate test data
y_pred_test = clf.predict(X_test_scaled)

# Add predicted labels to the separate test data DataFrame
test_df['mlp_outlier'] = y_pred_test

# Save the updated separate test data DataFrame back to its original file
test_df.reset_index(drop=True, inplace=True)  # Reset the index before saving, if necessary
test_df.to_feather(test_file_path)

print(f"Updated separate test data saved to {test_file_path}")

