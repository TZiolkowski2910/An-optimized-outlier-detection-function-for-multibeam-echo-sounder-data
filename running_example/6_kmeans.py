import os
import pandas as pd
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor

# Set the file directory for the k-means clustering output files
data_directory = '/data6/tziolkowski/data/raw_txt/running_example/chunks'

# Function to read and process each k-means clustering output file
def process_kmeans_output_file(file_path):
    # Read the k-means clustering output file into a DataFrame
    df = pd.read_feather(file_path)

    # Extract the columns used for k-means clustering
    X = df[['lat', 'lon', 'depth']]

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=4, random_state=0)
    df['kmeans_cluster'] = kmeans.fit_predict(X)

    # Calculate the distance from each data point to the center of its assigned cluster
    df['kmeans_distance'] = kmeans.transform(X).min(axis=1)

    # Mark soundings with a large distance as outliers
    df['kmeans_outlier'] = 0
    df.loc[df['kmeans_distance'] > df['kmeans_distance'].median(), 'kmeans_outlier'] = 1

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Save the processed file with the addition "kmeans" to the filename
    #output_file_path = file_path.replace('_lof.', '_kmeans.')
    df.to_feather(file_path)

    return df

# Define a function to process files in parallel
def process_files_parallel(file_paths):
    with ProcessPoolExecutor(max_workers=35) as executor:
        executor.map(process_kmeans_output_file, file_paths)

# Get a list of all k-means clustering output files in the directory
file_paths = [os.path.join(data_directory, file_name) for file_name in os.listdir(data_directory) if file_name.endswith('.feather')]

# Process the files in parallel
process_files_parallel(file_paths)
