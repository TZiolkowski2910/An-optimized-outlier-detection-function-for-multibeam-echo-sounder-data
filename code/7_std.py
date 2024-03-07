import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the files to be processed
data_directory = '/data6/tziolkowski/oqf/train3135/chunks'

# Function to read, process, and overwrite each file with STD analysis results
def process_std_output_file(file_path):
    # Read the file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the 'depth' column for STD analysis
    depth = df['depth']

    # Calculate the mean and standard deviation of the depth values
    mean = depth.mean()
    std = depth.std()

    # Calculate the threshold for outliers (3 standard deviations from the mean)
    threshold = mean + (3 * std)

    # Mark soundings with depth values higher than the threshold as outliers
    df['std_outlier'] = 0
    df.loc[depth > threshold, 'std_outlier'] = 1

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Overwrite the original file with the updated DataFrame
    df.to_feather(file_path)

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        # Execute the function for each file path in parallel
        executor.map(process_std_output_file, file_paths)

# Get a list of all relevant files in the directory
# Assuming you want to process all Feather files that have been processed previously (e.g., with KMeans)
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)

