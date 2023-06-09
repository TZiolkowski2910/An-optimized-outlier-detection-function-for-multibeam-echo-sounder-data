The following files are available:

    MSM88_accepted_example.txt: This file contains labelled multibeam data with only the accepted data points.
    MSM88_rejected_example.txt: This file contains labelled multibeam data with only the rejected data points.

To execute the scripts, use the corresponding number at the beginning of the file name. Make sure to adjust the paths in the scripts to access the correct files.

Steps 1-13 involve using the data from MSM88_accepted_example.txt and MSM88_rejected_example.txt to calculate outlier flags.

The Weighted function utilizes a subset of real-world data (chunk_14.6195--29.278_subset.feather) and the calculated outlier flags for the entire study area (refer to Section 3.2). It applies the proposed function (refer to Section 4.2) to this data.

Script Descriptions:

1_preprocessing.py: This script transforms raw multibeam data to the appropriate coordinate system (WGS84) and saves it as a feather file.
2_createChunks.py: Splits the data into chunks to reduce the data volume.
3_calculateFeatures.py: Calculates the following features for each point: "mean_depth_100m" (mean depth value of all soundings within a 100m radius), "normalized_distance_100m" (depth difference between the considered point and the average depth of all points within a 100m radius), and "std_deviation_depth_100m" (standard deviation of depths within a 100m radius of the considered point).

Outlier Detection Techniques:

The following algorithms can be directly applied to the example data using the created chunks:

4_dbscan.py: Density-based spatial clustering of applications with noise (DBSCAN).
5_iqr.py: Interquartile range (IQR) outlier detection.
6_kmeans.py: K-means clustering.
7_lof.py: Local Outlier Factor (LOF) algorithm.
8_mad.py: Median Absolute Deviation (MAD) outlier detection.
9_std.py: Standard deviation-based outlier detection.

For this example, we utilize pre-trained models: mlp_model.pkl (Multi-Layer Perceptron), random_forest_model.pkl (Random Forest), and logistic_regression_model.pkl (Logistic Regression). Since the data is limited, training models from scratch is not feasible. Before executing these three algorithms, the chunks need to be combined into a single file using:

10_combineChunks.py.

The following scripts can now be executed:

11_mlp.py
12_rf.py
13_lr.py

To calculate the dictionary, execute the script 14_calculate_dict_values.py. This will generate the file outlier_results.csv. Due to the limited data, some ranges may not have calculated values.

Weighted Function:

To demonstrate the weighted function, we use a subset of the real-world data (chunk_14.6195--29.278_subset.feather) along with the calculated dictionary values mentioned in the paper.

15_weighting_function.py: This script utilizes the calculated dictionary values and the best performing outlier detection techniques for each range. It calculates and adds the columns "f1_score_sum" and "weighted_score".

16_weighted_outlier_flag.py: This script sets the weighted outlier flag based on the "f1_score_sum" and "weighted_score" values calculated in 15_weighting_function.py.

17_testresult.py: This script provides statistics and data rows from the resulting file for evaluation purposes.
