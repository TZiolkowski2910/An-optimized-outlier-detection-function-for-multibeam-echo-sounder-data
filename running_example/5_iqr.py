import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the DBSCAN output files
data_directory = '/data6/tziolkowski/data/raw_txt/running_example/chunks'

# Function to read and process each DBSCAN output file
def process_dbscan_output_file(file_path):
    # Read the DBSCAN output file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the 'depth' column for IQR
    depth = df['depth']

    # Calculate the 25th percentile (Q1) and 75th percentile (Q3) of the 'depth' column
    q1 = depth.quantile(0.25)
    q3 = depth.quantile(0.75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Define a function to identify outliers based on IQR
    def is_outlier(row):
        return 1 if row['depth'] < q1 - 1.5 * iqr or row['depth'] > q3 + 1.5 * iqr else 0

    # Apply the function to each row in the DataFrame to identify outliers
    df['iqr_outlier'] = df.apply(is_outlier, axis=1)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Save the processed file with the addition "iqr" to the filename
   # output_file_path = file_path.replace('_mad.', '_iqr.')
    df.to_feather(file_path)

    return df

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        executor.map(process_dbscan_output_file, file_paths)

# Get a list of all DBSCAN output files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)
