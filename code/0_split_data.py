import pandas as pd

# Load the dataset
#data_path = '/data6/tziolkowski/PS139/chunks_10000_with_stats/cluster_statistic_features_PS139.feather'
data_path = "/data6/tziolkowski/mlp/with_stats/cluster_statistic_features_MSM88.feather"

df = pd.read_feather(data_path)

# Assuming 'Easting' here represents longitude (lon) for the purpose of this task
# Split the dataset based on Easting (lon) values
#train_df = df[(df['lat'] > -43.5) & (df['lat'] <= -25)]
#eval_df = df[(df['lat'] >= -48) & (df['lat'] <= -43.5)]

#train_df = df[(df['lat'] > -48) & (df['lat'] <= -31)]
#eval_df = df[(df['lat'] >= -31) & (df['lat'] <= -27.5)]

# Define eval_df for latitudes between -31 and -35
eval_df = df[(df['lat'] >= -35) & (df['lat'] < -31)]

# Define train_df for latitudes before -31 and after -35
train_df = df[(df['lat'] < -35) | (df['lat'] >= -31)]


# Information about the splits
print(f"Training dataset size: {len(train_df)}")
print(f"Evaluation dataset size: {len(eval_df)}")

# Save the split datasets to disk
train_data_path = '/data6/tziolkowski/oqf/train3135/OTD_features_train.feather'
eval_data_path = '/data6/tziolkowski/oqf/eval3135/OTD_features_eval.feather'

train_df.reset_index(drop=True).to_feather(train_data_path)
eval_df.reset_index(drop=True).to_feather(eval_data_path)

print(f"Training dataset saved to {train_data_path}")
print(f"Evaluation dataset saved to {eval_data_path}")

