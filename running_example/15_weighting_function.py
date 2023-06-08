import pandas as pd

# Define the dictionary
range_dict = {
    'mean_depth_100m': {
        '-6500|-6000': ('mlp_outlier', 0.780304272469669),
        '-6000|-5500': ('mlp_outlier', 0.7494152950277783),
        '-5500|-5000': ('mlp_outlier', 0.7469362561072259),
        '-5000|-4500': ('mlp_outlier', 0.7513027606123933),
        '-4500|-4000': ('mlp_outlier', 0.7439056226675391),
        '-4000|-3500': ('mlp_outlier', 0.7002059626869604),
        '-3500|-3000': ('mlp_outlier', 0.668807716855061),
        '-3000|-2500': ('mlp_outlier', 0.621471713967445),
        '-2500|-2000': ('mlp_outlier', 0.5869872524613132),
        '-2000|-1500': ('db_outlier', 0.0),
    },
    'normalized_distance_100m': {
        '-20|-14': ('db_outlier', 1.0),
        '-14|-11': ('kmeans_outlier', 1.0),
        '-11|-8': ('mlp_outlier', 0.991596638655),
        '-8|-5': ('mlp_outlier', 0.9912344777209),
        '-5|-2': ('mlp_outlier', 0.9798002512600),
        '-2|0': ('mlp_outlier', 0.75711573110566),
        '0|2': ('mlp_outlier', 0.690998676200591),
        '2|5': ('mlp_outlier', 0.974988440795032),
        '5|8': ('mlp_outlier', 0.991239048811013),
        '8|11': ('mlp_outlier', 1.0),
        '11|23': ('kmeans_outlier', 1.0),
    },
    'std_dev_depth_100m': {
        '0|100': ('mlp_outlier', 0.7392138986464022),
        '100|200': ('mlp_outlier', 0.8013254968406932),
        '200|300': ('lr_outlier', 0.849595451563525),
        '300|400': ('lr_outlier', 0.8345848977889029),
        '400|500': ('mad_outlier', 0.872276231981227),
        '500|600': ('mad_outlier', 0.8995535714285714),
        '600|inf': ('mad_outlier', 1.0),
    }
}

# Load the dataset
df = pd.read_feather('/data6/tziolkowski/oqf/correct/grids/statisticbased/grids/chunk_14.6195--29.278_subset.feather')

# Define a function to calculate the weighted score for each row
def calculate_weighted_score(row):
    # Initialize the weighted score for the row
    weighted_score = 0

    # Initialize the sum of f1 scores for the row
    f1_score_sum = 0

    # Loop over the three columns
    for col in ['mean_depth_100m', 'normalized_distance_100m', 'std_dev_depth_100m']:

        # Extract the value of the column for the current row
        val = row[col]

        # Find the range in the dictionary to which the value belongs
        for key in range_dict[col]:
            range_min, range_max = key.split('|')
            if float(range_min) <= val < float(range_max):
                # Multiply the value with the f1 score and add it to the weighted score
                outlier_type, f1_score = range_dict[col][key]
                weighted_score += row[outlier_type] * f1_score
                f1_score_sum += f1_score
                break

    # Add the weighted score and the sum of f1 scores to the row
    row['weighted_score'] = weighted_score
    row['f1_score_sum'] = f1_score_sum

    # Return the updated row
    return row

# Apply the function to each row of the data frame
df = df.apply(calculate_weighted_score, axis=1)

# Save the updated data frame to file
df.to_feather('/data6/tziolkowski/oqf/correct/grids/statisticbased/grids/chunk_14.6195--29.278_subset_weighted.feather')
