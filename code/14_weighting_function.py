import pandas as pd

# Define the dictionary
#range_dict = {
#    'mean_depth_100m': {
#        '-6500|-6000': ('mlp_outlier', 0.780304272469669),
#        '-6000|-5500': ('mlp_outlier', 0.7494152950277783),
#        '-5500|-5000': ('mlp_outlier', 0.7469362561072259),
#        '-5000|-4500': ('mlp_outlier', 0.7513027606123933),
#        '-4500|-4000': ('mlp_outlier', 0.7439056226675391),
#        '-4000|-3500': ('mlp_outlier', 0.7002059626869604),
#        '-3500|-3000': ('mlp_outlier', 0.668807716855061),
#        '-3000|-2500': ('mlp_outlier', 0.621471713967445),
#        '-2500|-2000': ('mlp_outlier', 0.5869872524613132),
#        '-2000|-1500': ('db_outlier', 0.0),
#    },
#    'normalized_distance_100m': {
#        '-20|-14': ('db_outlier', 1.0),
#        '-14|-11': ('kmeans_outlier', 1.0),
#        '-11|-8': ('mlp_outlier', 0.991596638655),
#        '-8|-5': ('mlp_outlier', 0.9912344777209),
#        '-5|-2': ('mlp_outlier', 0.9798002512600),
#        '-2|0': ('mlp_outlier', 0.75711573110566),
#        '0|2': ('mlp_outlier', 0.690998676200591),
#        '2|5': ('mlp_outlier', 0.974988440795032),
#        '5|8': ('mlp_outlier', 0.991239048811013),
#        '8|11': ('mlp_outlier', 1.0),
#        '11|23': ('kmeans_outlier', 1.0),
#    },
#    'std_dev_depth_100m': {
#        '0|100': ('mlp_outlier', 0.7392138986464022),
#        '100|200': ('mlp_outlier', 0.8013254968406932),
#        '200|300': ('lr_outlier', 0.849595451563525),
#        '300|400': ('lr_outlier', 0.8345848977889029),
#        '400|500': ('mad_outlier', 0.872276231981227),
#        '500|600': ('mad_outlier', 0.8995535714285714),
#        '600|inf': ('mad_outlier', 1.0),
#    }
#}


#range_dict = {
#    'mean_depth_100m': {
#        '-6500|-6000': ('lr_outlier', 1.000000),
#        '-6000|-5500': ('mlp_outlier', 0.772605),
#        '-5500|-5000': ('mlp_outlier', 0.707848),
#        '-5000|-4500': ('db_outlier', 0),
#        '-4500|-4000': ('mlp_outlier', 0),
#        '-4000|-3500': ('mlp_outlier', 0),
#        '-3500|-3000': ('mlp_outlier', 0),
#        '-3000|-2500': ('mlp_outlier', 0),
#        '-2500|-2000': ('mlp_outlier', 0),
#        '-2000|-1500': ('db_outlier', 0),
#    },
#    'normalized_distance_100m': {
#        '-20|-14': ('db_outlier', 0),
#        '-14|-11': ('db_outlier', 0),
#        '-11|-8': ('kmeans_outlier', 1.0),
#        '-8|-5': ('mlp_outlier', 1.0),
#        '-5|-2': ('mlp_outlier', 0.978385),
#        '-2|0': ('mlp_outlier', 0.743540),
#        '0|2': ('mlp_outlier', 0.635423),
#        '2|5': ('mlp_outlier', 0.976099),
#        '5|8': ('mlp_outlier', 1.0),
#        '8|11': ('kmeans_outlier', 1.0),
#        '11|23': ('db_outlier', 0.0),
#    },
#    'std_dev_depth_100m': {
#        '0|100': ('mlp_outlier', 0.728813),
#        '100|200': ('mlp_outlier', 0.887107),
#        '200|300': ('mlp_outlier', 0.832978),
#        '300|400': ('lr_outlier', 0.889488),
 #       '400|500': ('lr_outlier', 0.966292),
 #       '500|600': ('lr_outlier', 1.0),
  #      '600|inf': ('kmeans_outlier', 1.0),
  #  }
#}


range_dict = {
    'mean_depth_100m': {
        '-6500|-6000': ('mlp_outlier', 0.773523),
        '-6000|-5500': ('mlp_outlier', 0.735672),
        '-5500|-5000': ('mlp_outlier', 0.741361),
        '-5000|-4500': ('mlp_outlier', 0.777452),
        '-4500|-4000': ('kmeans_outlier', 0.910140),
        '-4000|-3500': ('mlp_outlier', 0),
        '-3500|-3000': ('mlp_outlier', 0),
        '-3000|-2500': ('mlp_outlier', 0),
        '-2500|-2000': ('mlp_outlier', 0),
        '-2000|-1500': ('db_outlier', 0),
    },
    'normalized_distance_100m': {
        '-20|-14': ('db_outlier', 0),
        '-14|-11': ('db_outlier', 0),
        '-11|-8': ('kmeans_outlier', 1.0),
        '-8|-5': ('mlp_outlier', 1.0),
        '-5|-2': ('mlp_outlier', 0.983316),
        '-2|0': ('mlp_outlier', 0.752514),
        '0|2': ('mlp_outlier', 0.695395),
        '2|5': ('mlp_outlier', 0.979072),
        '5|8': ('mlp_outlier', 1.0),
        '8|11': ('db_outlier', 1.0),
        '11|23': ('db_outlier', 0.0),
    },
    'std_dev_depth_100m': {
        '0|100': ('mlp_outlier', 0.740580),
        '100|200': ('mlp_outlier', 0.813760),
        '200|300': ('mlp_outlier', 0.860916),
        '300|400': ('lr_outlier', 0.921778),
        '400|500': ('lr_outlier', 0.954315),
        '500|600': ('lr_outlier', 0.989247),
        '600|inf': ('db_outlier', 0),
    }
}


# Load the dataset
#df = pd.read_feather('/data6/tziolkowski/oqf/eval/mlp_with_predictions_spatialsplit.feather')
file_path = '/data6/tziolkowski/oqf/eval3135/chunks/OTD_features_eval.feather'
df = pd.read_feather(file_path)


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
#df.to_feather('/data6/tziolkowski/oqf/eval/mlp_with_predictions_spatialsplit_weighted.feather')
df.to_feather(file_path)
