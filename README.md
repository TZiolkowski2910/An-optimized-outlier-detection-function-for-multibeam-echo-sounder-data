# Towards-an-Outlier-Quantification-Framework-for-multibeam-Echo-sounder-data

This repository contains two main directories: "code" and "running example."

    Code Directory:
        This directory contains the code used in the paper. It includes scripts and files necessary to replicate the experiments and implement the proposed methods.

    Running Example Directory:
        Within this directory, you will find a running example that demonstrates the application of the code.
        Due to the limited size of the running example data (only 10,000 rows), the machine learning-based code has been modified. Instead of training models on this small dataset, pre-trained models from the           real-world example are utilized.
        It is important to note that there are no differences in the running example code itself; only the approach to training the models has been adjusted.
        Additionally, there is a separate readme.md file in the running example directory. This readme provides a manual on how to use the code and offers a description of the data used in the running example.
        It is possible to selectively execute the scripts 15_weighting_function.py and 16_weighted_outlier_flag.py to evaluate the proposed weighting function independently. These scripts specifically utilize           the data from chunk_14.6195--29.278_subset.feather for analysis and calculations.

Please note: The code for calculating additional features and outlier labels incorporates multi-processing techniques to enhance efficiency and speed during the process.
