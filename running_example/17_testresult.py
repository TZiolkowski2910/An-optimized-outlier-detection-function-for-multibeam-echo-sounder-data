import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

# Path to the combined features file
#combined_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/grids/combined_predictions_weightedscore_10.feather'

#combined_file_path = '/data6/tziolkowski/data/raw_txt/running_example/combinedFeatherOutlier'
combined_file_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/grids/chunk_14.6195--29.278_subset_weighted.feather'

# Load the combined features DataFrame
df = pd.read_feather(combined_file_path)

#print(df['weighted_outlier'].value_counts())


# Display column names
print("Column Names:")
print(df.columns)
print(len(df))
# Display first 5 rows of the DataFrame
print("\nFirst 5 Rows:")
print(df.head())

# Display first 5 rows of the DataFrame
#print("\nLast 5 Rows:")
#print(df.tail())



# Display minimum and maximum values of each column
#print("\nMinimum and Maximum Values:")
#for column in df.columns:
#    print(f"{column}: Min={df[column].min()}, Max={df[column].max()}"

