import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib  # for saving the model

# Load the training data from Feather file
train_file_path = '/data6/tziolkowski/oqf/train3135/chunks/OTD_features_train.feather'
train_data = pd.read_feather(train_file_path)

# Drop NaN values
train_data = train_data.dropna()

# Specify the columns to be used as features and target variable
X = train_data[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y = train_data['outlier']

# Split the training data into training and evaluation sets (80% train, 20% eval)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression classifier
clf = LogisticRegression(random_state=42)

# Fit the classifier to the training part of the data
clf.fit(X_train, y_train)

# Predict labels for the evaluation part of the data
y_pred_eval = clf.predict(X_eval)

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
test_data = pd.read_feather(test_file_path)

# Drop NaN values in test data
test_data = test_data.dropna()

# Specify the columns to be used as features for the test data
X_test = test_data[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]

# Predict labels for the separate test data
y_pred_test = clf.predict(X_test)

# Add predicted labels to the separate test data DataFrame
test_data['lr_outlier'] = y_pred_test

# Reset the index before saving, if necessary
test_data.reset_index(drop=True, inplace=True)

# Save the updated separate test data DataFrame back to its original file
test_data.to_feather(test_file_path)

print(f"Updated separate test data saved to {test_file_path}")

