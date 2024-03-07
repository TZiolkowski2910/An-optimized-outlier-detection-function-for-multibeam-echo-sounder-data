import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the processed files
data_directory = '/data6/tziolkowski/oqf/eval3135/chunks'

# Function to read, process each file, and save it back
def process_file(file_path):
    # Read the file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the columns used for LOF
    X = df[['lat', 'lon', 'depth']]

    # Perform LOF for outlier detection
    lof = LocalOutlierFactor(n_neighbors=82, contamination=0.1)
    df['lof_outlier'] = lof.fit_predict(X)

    # Set LOF outlier values to 0 for no outlier and 1 for outlier
    df['lof_outlier'] = (df['lof_outlier'] == -1).astype(int)

    # Reset the index before saving
    df.reset_index(drop=True, inplace=True)

    # Overwrite the original file
    df.to_feather(file_path)

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        executor.map(process_file, file_paths)

# Get a list of all relevant files in the directory (assuming you want to process all Feather files)
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)

