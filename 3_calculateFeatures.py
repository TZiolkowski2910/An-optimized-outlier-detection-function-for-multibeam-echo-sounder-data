import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing as mp
import os
import shutil

# Define the radius in meters
radius = 100

# Function to calculate statistics for a chunk
def calculate_chunk_stats(chunk_path):
    # Load the chunk
    df = pd.read_feather(chunk_path)

    # Calculate the Euclidean distances between all soundings
    distances = cdist(df[['lat', 'lon']], df[['lat', 'lon']])

    # Calculate the square of the radius in degrees
    radius_degrees = radius / (111.32 * 1000)

    # Find all soundings within the radius for each sounding
    neighbors = [np.where(distances[i] <= radius_degrees)[0] for i in range(len(df))]

    # Calculate the standard deviation of depth in the neighborhood for each sounding
    std_dev_depth = [df['depth'][neighbors[i]].std() for i in range(len(df))]

    # Calculate the mean depth in the neighborhood for each sounding
    mean_depth = [df['depth'][neighbors[i]].mean() for i in range(len(df))]

    # Calculate the normalized distance from the depth of the sounding to the standard deviation at 100m
    normalized_distance = [(df['depth'][i] - mean_depth[i]) / std_dev_depth[i] if std_dev_depth[i] != 0 else 0 for i in range(len(df))]

    # Add the calculated statistics to the dataframe
    df['std_dev_depth_100m'] = std_dev_depth
    df['mean_depth_100m'] = mean_depth
    df['normalized_distance_100m'] = normalized_distance

    # Save the updated chunk to a new Feather file in the 'with_stats' directory
    stats_dir = '/data6/tziolkowski/PS139/chunks_10000_with_stats'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    stats_file_path = os.path.join(stats_dir, os.path.basename(chunk_path).replace('.feather', '_with_stats.feather'))
    df.to_feather(stats_file_path)

    # Move the processed chunk file to the 'processed' directory
    processed_dir = '/data6/tziolkowski/PS139/chunks_10000_chunks_processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    shutil.move(chunk_path, os.path.join(processed_dir, os.path.basename(chunk_path)))

# Directory where chunks are saved
#chunk_dir = '/data6/tziolkowski/mlp/chunks'
chunk_dir = "/data6/tziolkowski/PS139/chunks_10000"

# Get the paths of all chunks
chunk_paths = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith('.feather')]

# Use multiprocessing to calculate statistics for each chunk in parallel
with mp.Pool(processes=4) as pool:
    pool.map(calculate_chunk_stats, chunk_paths)

