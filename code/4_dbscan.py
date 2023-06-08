import os
import pandas as pd
from sklearn.cluster import DBSCAN
from concurrent.futures import ProcessPoolExecutor

# Set the file directory and parameters for DBSCAN
data_directory = '/data6/tziolkowski/oqf/data2/grids/'
eps = 0.1
min_samples = 6

# Function to read and process each chunk file
def process_chunk_file(file_path):
    # Read the chunk file into a DataFrame
    df = pd.read_feather(file_path)

    # Perform DBSCAN for outlier detection
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['labels'] = dbscan.fit_predict(df[['lat', 'lon', 'depth']])

    # Add a column "db_outlier" which contains 1 for detected outliers and 0 for no outlier
    df['db_outlier'] = 0
    df.loc[df['labels'] == -1, 'db_outlier'] = 1

    # Remove the "labels" column
    df.drop(columns=['labels'], inplace=True)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Save the processed chunk file with the addition "db" to the filename
    output_file_path = file_path.replace('.feather', '_db.feather')
    df.to_feather(output_file_path)

    return df

# Iterate over all chunk files in the directory and apply DBSCAN in parallel with 35 CPUs
with ProcessPoolExecutor(max_workers=5) as executor:
    # Get the file paths of all chunk files
    file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

    # Process the chunk files in parallel
    dfs = executor.map(process_chunk_file, file_paths)
