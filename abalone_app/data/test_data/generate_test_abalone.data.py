#!/usr/bin/env python
# coding: utf-8

"""
This script is to generate all expected output from the units test.
Run the script only if you have detected a error and fix it.
"""
import pickle
import pandas as pd
from abalone_prediction import modeling as md


# 0/ Creata a short pandas dataframe from the original data
path = '../abalone.data'
df = pd.read_csv(path, header=None)
df = df.iloc[:10, ]
df.to_csv("test_abalone.data", header=None, index=False)

# 1. Load Data
path = "test_abalone.data"
initial_df_expected = md.load_data_abalone(path)

# 2. Feature Preparation
data_splitted_expected = md.split_data(initial_df_expected)
data_preprocessed_expected = md.preprocessing_data(data_splitted_expected)

# 3. Training
model_expected = md.train_model(data_preprocessed_expected)

# 4. Prediction
my_abalone = {
    "sex": "F",
    "length": 0.815000,
    "diameter": 1.055000,
    "height": 1.130000,
    "whole_weight": 2.825500,
    "shucked_weight": 1.488000,
    "viscera_weight": 1.760000,
    "shell_weight": 0.001500
}

prediction_expected = md.predict_abalone(my_abalone, model_expected)

# Saving variables in file
variables = [initial_df_expected,
             data_splitted_expected,
             data_preprocessed_expected,
             model_expected,
             prediction_expected]

savefile = open("all_data_test", "wb")
pickle.dump(variables, savefile)
savefile.close()
