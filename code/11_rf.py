import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib  # for saving the model

# Load the train and test DataFrames from Feather files
train_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/train_data.feather'
test_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/test_data.feather'
df_train = pd.read_feather(train_file_path)
df_test = pd.read_feather(test_file_path)

# Specify the columns to be used as features and target variable
X_train = df_train[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y_train = df_train['outlier']
X_test = df_test[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]

# Create a Random Forest classifier with specified parameters
n_estimators = 40
max_depth = 5
clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Save the trained model to file
model_file_path = '/data6/tziolkowski/oqf/correct/grids/model/random_forest_model.pkl'
joblib.dump(clf, model_file_path)

# Predict labels for test data
y_pred = clf.predict(X_test)

# Calculate precision, recall, and F1 score
precision = precision_score(df_test['outlier'], y_pred)
recall = recall_score(df_test['outlier'], y_pred)
f1 = f1_score(df_test['outlier'], y_pred)

# Print the results
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))

# Add predicted labels to test DataFrame
df_test['predicted_outlier'] = y_pred

# Save test DataFrame with predicted labels to a new feather file
predicted_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/test_data_predicted.feather'
df_test.to_feather(predicted_file_path)
