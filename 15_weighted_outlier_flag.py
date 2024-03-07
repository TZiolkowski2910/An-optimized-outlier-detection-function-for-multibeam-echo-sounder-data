import pandas as pd

# Load the Feather file
#file_path = '/data6/tziolkowski/oqf/eval/mlp_with_predictions_spatialsplit_weighted.feather'
file_path = '/data6/tziolkowski/oqf/eval3135/chunks/OTD_features_eval.feather'
df = pd.read_feather(file_path)

# Add the 'weighted_outlier' column based on the specified conditions
df['weighted_outlier'] = ((df['weighted_score'] != 0) & (df['weighted_score'] == df['f1_score_sum'])).astype(int)

# Save the updated DataFrame as a Feather file
#output_file_path = '/data6/tziolkowski/oqf/eval4748/chunks/OTD_features_eval_weighted_updated.feather'
df.to_feather(file_path)
