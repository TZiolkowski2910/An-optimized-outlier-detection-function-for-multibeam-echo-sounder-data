import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the DBSCAN output files
data_directory = '/data6/tziolkowski/oqf/data2/grids/statisticbased/'

# Function to read and process each DBSCAN output file
def process_dbscan_output_file(file_path):
    # Read the DBSCAN output file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the 'depth' column for MAD
    depth = df['depth']

    # Calculate the median of the 'depth' column
    median = depth.median()

    # Calculate the absolute deviation from the median
    abs_deviation = (depth - median).abs()

    # Calculate the median of the absolute deviations
    mad = abs_deviation.median()

    # Define a function to identify outliers based on MAD
    def is_outlier(row):
        return 1 if abs(row['depth'] - median) > 3 * mad else 0

    # Apply the function to each row in the DataFrame to identify outliers
    df['mad_outlier'] = df.apply(is_outlier, axis=1)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Save the processed file with the addition "mad" to the filename
    output_file_path = file_path.replace('_std.', '_mad.')
    df.to_feather(output_file_path)

    return df

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        executor.map(process_dbscan_output_file, file_paths)

# Get a list of all DBSCAN output files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('_std.feather')]

# Process the files in parallel
process_files_parallel(file_paths)

