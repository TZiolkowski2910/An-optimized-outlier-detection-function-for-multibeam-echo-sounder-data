import os
import pandas as pd
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the files to be processed
data_directory = '/data6/tziolkowski/oqf/train3135/chunks'

# Function to read, process, and overwrite each file with KMeans clustering results
def process_kmeans_output_file(file_path):
    # Read the file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the columns used for KMeans clustering
    X = df[['lat', 'lon', 'depth']]

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=0)
    df['kmeans_cluster'] = kmeans.fit_predict(X)

    # Calculate the distance from each data point to the center of its assigned cluster
    df['kmeans_distance'] = kmeans.transform(X).min(axis=1)

    # Mark soundings with a large distance as outliers
    df['kmeans_outlier'] = 0
    df.loc[df['kmeans_distance'] > df['kmeans_distance'].median(), 'kmeans_outlier'] = 1

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Overwrite the original file with the updated DataFrame
    df.to_feather(file_path)

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        # Execute the function for each file path in parallel
        executor.map(process_kmeans_output_file, file_paths)

# Get a list of all relevant files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)

