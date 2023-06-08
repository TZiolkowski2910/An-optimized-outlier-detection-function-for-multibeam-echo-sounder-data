import pandas as pd
import os

# set the directory path where the files are located
dir_path = '/data6/tziolkowski/data/raw_txt/running_example/chunks'

# get the list of file names that match the pattern
file_names = [f for f in os.listdir(dir_path) if f.startswith('chunk_')]

# sort the file names in ascending order
file_names.sort()

# create an empty dataframe
combined_df = pd.DataFrame()

# iterate over the files and concatenate them into a single dataframe
for file_name in file_names:
    file_path = os.path.join(dir_path, file_name)
    df = pd.read_feather(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# save the combined dataframe as a feather file
combined_df.to_feather('/data6/tziolkowski/data/raw_txt/running_example/combinedFeatherOutlier')
