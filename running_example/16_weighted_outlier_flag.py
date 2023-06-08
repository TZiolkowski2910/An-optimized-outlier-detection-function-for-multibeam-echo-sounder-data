import pandas as pd
import os

# set the directory path where the files are located
dir_path = '/data6/tziolkowski/oqf/correct/grids/statisticbased/grids/'

# get the list of file names that match the pattern
file_names = [f for f in os.listdir(dir_path) if f.startswith('chunk_14.6195--29.278_subset_')]

# sort the file names in ascending order
file_names.sort()

# create an empty dataframe
combined_df = pd.DataFrame()

# iterate over the files and concatenate them into a single dataframe
for file_name in file_names:
    file_path = os.path.join(dir_path, file_name)
    df = pd.read_feather(file_path)
    
    # apply the calculation to add the new column "weighted_outlier"
    df['weighted_outlier'] = df.apply(lambda row: 1 if row['weighted_score']!=0 and row['weighted_score']==row['f1_score_sum'] else 0 if row['weighted_score']==0 else 2, axis=1)
    
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# save the combined dataframe as a feather file
combined_df.to_feather('/data6/tziolkowski/oqf/correct/grids/statisticbased/grids/chunk_14.6195--29.278_subset_weighted.feather')

# count the number of rows with each value in the "weighted_outlier" column
count = combined_df['weighted_outlier'].value_counts()

# print the counts
print(count)
