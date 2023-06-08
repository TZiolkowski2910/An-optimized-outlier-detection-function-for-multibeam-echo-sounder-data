import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the STD output files
data_directory = '/data6/tziolkowski/data/raw_txt/running_example/chunks'

# Function to read and process each STD output file
def process_std_output_file(file_path):
    # Read the STD output file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the 'depth' column for STD
    depth = df['depth']

    # Calculate the mean and standard deviation of the depth values
    mean = depth.mean()
    std = depth.std()

    # Calculate the threshold for outliers
    threshold = mean + (3 * std)

    # Mark soundings with depth values higher than the threshold as outliers
    df['std_outlier'] = 0
    df.loc[depth > threshold, 'std_outlier'] = 1

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Save the processed file with the addition "std" to the filename
    #output_file_path = file_path.replace('_kmeans.', '_std.')
    df.to_feather(file_path)

    return df

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        executor.map(process_std_output_file, file_paths)

# Get a list of all STD output files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)
