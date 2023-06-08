import os
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the DBSCAN output files
data_directory = '/data6/tziolkowski/data/raw_txt/running_example/chunks'

# Function to read and process each DBSCAN output file
def process_dbscan_output_file(file_path):
    # Read the DBSCAN output file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the columns used for LOF
    X = df[['lat', 'lon', 'depth']]

    # Perform LOF for outlier detection
    lof = LocalOutlierFactor(n_neighbors=82, contamination=0.1)
    df['lof_outlier'] = lof.fit_predict(X)

    # Set LOF outlier values to 0 for no outlier and 1 for outlier
    df['lof_outlier'] = (df['lof_outlier'] == -1).astype(int)
    
    df.reset_index(drop=True, inplace=True)

    # Save the processed file with the addition "lof" to the filename
    #output_file_path = file_path.replace('_db.', '_lof.')
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
