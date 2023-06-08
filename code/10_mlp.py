import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # for saving the model

# Load the train and test data from Feather files
train_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/train_data.feather'
test_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/test_data.feather'
train_df = pd.read_feather(train_file_path)
test_df = pd.read_feather(test_file_path)

# Specify the columns to be used as features and target variable
X_train = train_df[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y_train = train_df['outlier']
X_test = test_df[['lat', 'lon', 'depth', 'std_dev_depth_100m', 'mean_depth_100m', 'normalized_distance_100m']]
y_test = test_df['outlier']

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(32, 32, 32), activation='relu', solver='sgd', batch_size=32,
                    learning_rate='adaptive', max_iter=50, early_stopping=True, tol=0.002, random_state=42,
                    validation_fraction=0.2, n_iter_no_change=3, verbose=1)

# Fit the classifier to the training data
clf.fit(X_train_scaled, y_train)

# Save the trained model to file
model_file_path = '/data6/tziolkowski/oqf/correct/grids/model/mlp_model.pkl'
joblib.dump(clf, model_file_path)

# Predict labels for test data
y_pred = clf.predict(X_test_scaled)

# Add predicted labels to test feather file
test_df['prediction'] = y_pred
output_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/test_data_mlp_with_predictions.feather'
test_df.to_feather(output_file_path)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
