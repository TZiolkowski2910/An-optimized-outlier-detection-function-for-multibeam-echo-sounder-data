import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the files to be processed
data_directory = '/data6/tziolkowski/oqf/train3135/chunks'

# Function to read, process, and overwrite each file with IQR analysis results
def process_iqr_output_file(file_path):
    # Read the file into a DataFrame
    df = pd.read_feather(file_path)

    # Calculate the Interquartile Range (IQR) for outlier detection
    q1 = df['depth'].quantile(0.25)
    q3 = df['depth'].quantile(0.75)
    iqr = q3 - q1

    # Identify outliers based on IQR
    df['iqr_outlier'] = ((df['depth'] < (q1 - 1.5 * iqr)) | (df['depth'] > (q3 + 1.5 * iqr))).astype(int)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Overwrite the original file with the updated DataFrame
    df.to_feather(file_path)

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        # Execute the function for each file path in parallel
        executor.map(process_iqr_output_file, file_paths)

# Get a list of all relevant files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)

