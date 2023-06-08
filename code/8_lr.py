import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib  # for saving the model

# Load the train and test data from Feather files
train_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/train_data.feather'
test_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/test_data.feather'

train_data = pd.read_feather(train_file_path)
test_data = pd.read_feather(test_file_path)

# Specify the columns to be used as features and target variable
X_train = train_data[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y_train = train_data['outlier']

X_test = test_data[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y_test = test_data['outlier']

# Create a Logistic Regression classifier
clf = LogisticRegression(random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Save the trained model to file
model_file_path = '/data6/tziolkowski/oqf/correct/grids/model/logistic_regression_model.pkl'
joblib.dump(clf, model_file_path)

# Predict labels for test data
y_pred = clf.predict(X_test)

# Add predicted labels to the test data DataFrame
test_data['predicted_outlier'] = y_pred

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Save the test data DataFrame with predicted labels to file
output_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/test_data_lr_with_predictions.feather'
test_data.to_feather(output_file_path)
