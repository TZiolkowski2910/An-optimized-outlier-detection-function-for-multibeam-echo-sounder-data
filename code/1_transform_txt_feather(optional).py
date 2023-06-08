import os
import pandas as pd
import pyproj

# Directory paths for accepted and rejected data
accepted_dir = '/data6/tziolkowski/oqf/data2/AREA4_process/Accepted_data'
rejected_dir = '/data6/tziolkowski/oqf/data2/AREA4_process/Rejected_data'

# Coordinate transformation function
def transform_coordinates(lon, lat):
    # Define source and target coordinate systems
    source_proj = pyproj.Proj(proj='utm', zone=27, ellps='WGS84')
    target_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    
    # Perform coordinate transformation
    lat, lon = pyproj.transform(source_proj, target_proj, lat, lon)
    
    return lat, lon

# Read all txt files in accepted directory and concatenate into a single DataFrame
accepted_files = [os.path.join(accepted_dir, f) for f in os.listdir(accepted_dir) if f.endswith('.txt')]
df_accepted = pd.concat((pd.read_csv(file, names=['lat', 'lon', 'depth']) for file in accepted_files), ignore_index=True)

# Read all txt files in rejected directory and concatenate into a single DataFrame
rejected_files = [os.path.join(rejected_dir, f) for f in os.listdir(rejected_dir) if f.endswith('.txt')]
df_rejected = pd.concat((pd.read_csv(file, names=['lat', 'lon', 'depth']) for file in rejected_files), ignore_index=True)

# Concatenate accepted and rejected DataFrames into a single DataFrame
df = pd.concat([df_accepted, df_rejected], ignore_index=True)

# Transform lat and lon columns to WGS84 coordinates
df['lat'], df['lon'] = transform_coordinates(df['lon'], df['lat'])

# Add "outlier" column with values 0 for accepted data and 1 for rejected data
df['outlier'] = 0
df.loc[df.index.isin(df_rejected.index), 'outlier'] = 1

# Reset index
df.reset_index(drop=True, inplace=True)

# Save the concatenated DataFrame as a feather file
feather_file = '/data6/tziolkowski/oqf/data2/concatenated_area4.feather'
df.to_feather(feather_file)

print("Concatenated data with outlier column, column renaming, and coordinate transformation saved as a feather file: ", feather_file)
