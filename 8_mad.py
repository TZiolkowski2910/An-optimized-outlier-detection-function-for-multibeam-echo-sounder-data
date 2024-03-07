import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the files to be processed
data_directory = '/data6/tziolkowski/oqf/train3135/chunks'

# Function to read, process, and overwrite each file with MAD analysis results
def process_mad_output_file(file_path):
    # Read the file into a DataFrame
    df = pd.read_feather(file_path)

    # Calculate the Median Absolute Deviation (MAD) based outlier detection
    median = df['depth'].median()
    mad = abs(df['depth'] - median).median()
    threshold = 3 * mad

    # Define outliers based on MAD
    df['mad_outlier'] = ((abs(df['depth'] - median) > threshold) * 1)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Overwrite the original file with the updated DataFrame
    df.to_feather(file_path)

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        # Execute the function for each file path in parallel
        executor.map(process_mad_output_file, file_paths)

# Get a list of all relevant files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)

