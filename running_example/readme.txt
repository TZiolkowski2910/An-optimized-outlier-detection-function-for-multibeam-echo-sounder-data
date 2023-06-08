MSM88_accepted_example.txt: Labelled multibeam data, contains only the accepted data points
MSM88_rejected_example.txt: Labelled multibeam data, contains only the rejected data points

1_preprocessing.py = Transform raw multibeam data to right coordinate system and feather file
2_createChunks = Split the data in chunks to reduce the data volume
3_calculateFeatures.py = Calculate features "mean\_depth\_100m" (the mean of the depth values of all soundings within 100m radius of the point being considered), "normalized\_distance\_100m" (the difference in depth between point being considered on the average depth of all points within 100m radius) and "std\_deviation\_depth\_100m" (the standard deviation of the depths of all points within 100m radius of the point being considered.)

Outlier Detection Techniques:
The statistic-based and distance/ density-based algorithms can be executed directly on the example data using the created chunks:
4_dbscan.py
5_iqr.py
6_kmeans.py
7_lof.py
8_mad.py
9_std.py

For this running example, we use the trained mlp (mlp_model.pkl), random forest (random_forest_model.pkl) and logistic regression model (logistic_regression_model.pkl). It does not make sense to train a model on such little data. Before executing these three algorithms the chunks have to be combined in one file again with:
10_combineChunks.py.

Now the following three scripts can be executed:
11_mlp.py
12_rf.py
13_lr.py

For calculating the dictionary, the script 14_calculate_dict_values.py can now be executed. As a result the file outlier_results.csv is created. Due to less data, there is no calculated value for each range.

Weighted Function:
For example of the weighted function, a subset of the real world data (chunk_14.6195--29.278_subset.feather) is used with the calculated dictionary values presented in the paper.
15_weighting_function.py: Uses the calculated dictionary values and the best performing outlier detection techniques for each range. As a result, the columns f1_score_sum and weighted_score are calculated and added
16_weighted_outlier_flag.py: Sets the weighted outlier flag depending on the f1_Score_sum and weighted_score calculated in 4_weighting_function.py

17_testresult.py: Prints out some statistics and data rows from the resulting file for evaluation.