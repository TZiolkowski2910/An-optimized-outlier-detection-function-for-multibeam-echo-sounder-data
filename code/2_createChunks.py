import pandas as pd
import numpy as np
import os

# Load the multibeam data set into a pandas DataFrame
#df = pd.read_feather('/data6/tziolkowski/subset_output.feather')  # Replace 'multibeam_data.csv' with your file name
# Specify the column names in a list
#column_names = ['lon', 'lat', 'depth']

#df = pd.read_csv('/data6/tziolkowski/PS139/PS139_concatenated_data.xyz')

df = pd.read_feather("/data6/tziolkowski/oqf/eval3135/OTD_features_eval.feather")

# Define the size of each chunk in meters
chunk_size_m = 10000

# Convert latitude and longitude values to meters using a conversion factor (approximate)
lat_to_m = 111320  # 1 degree of latitude is approximately 111,320 meters
lon_to_m = 111320 * np.cos(np.deg2rad(df['lat'].mean()))  # 1 degree of longitude is approximately 111,320 meters times the cosine of the latitude

# Calculate the number of chunks in latitude and longitude directions
num_chunks_lat = int(np.ceil((df['lat'].max() - df['lat'].min()) / (chunk_size_m / lat_to_m)))
num_chunks_lon = int(np.ceil((df['lon'].max() - df['lon'].min()) / (chunk_size_m / lon_to_m)))

# Group the data by latitude and longitude values to form chunks
grouped = df.groupby([pd.cut(df['lat'], num_chunks_lat, retbins=True)[0], pd.cut(df['lon'], num_chunks_lon, retbins=True)[0]])

# Directory to save the chunks
#output_dir = '/data6/tziolkowski/PS139/chunks_10000'
output_dir = '/data6/tziolkowski/oqf/eval3135/chunks'

# Save each chunk to disk
for name, group in grouped:
    # Extract the latitude and longitude ranges for the chunk
    lat_range = name[0].mid
    lon_range = name[1].mid
    
    # Create a unique chunk ID based on the latitude and longitude ranges
    chunk_id = f'chunk_{lat_range}-{lon_range}'

    group = group.reset_index(drop=True)

    # Save the chunk to disk in the specified directory
    output_path = os.path.join(output_dir, f'{chunk_id}.feather')
    group.to_feather(output_path)  # Replace '.csv' with your desired file format
    print(f'Saved {chunk_id} to disk at {output_path}.')
