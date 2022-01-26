#!/usr/bin/env python
# coding: utf-8
import pickle
import os.path
from abalone_prediction import modeling as md
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier



# Load the expected data for the units tests
expected_data_path="../data/test_data/all_data_test"
if os.path.isfile(expected_data_path):
    savefile = open(expected_data_path,"rb")
    variables = pickle.load(savefile)
    savefile.close()
else:
    raise TypeError("Need file 'all_data_test' to load the expected values.")

initial_df_expected = variables[0]
data_splitted_expected = variables[1]
data_preprocessed_expected = variables[2]
model_expected = variables[3]
prediction_expected = variables[4]
path = "../data/test_data/test_abalone.data"


def test_load_data_abalone():
    output = md.load_data_abalone(path)

    # Expected
    names = ['sex', 'length', 'diameter', 'height', 'whole_weight',
             'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    df = pd.read_csv(path, header=None, names=names)

    df['target'] = (df['rings'] >= 10).astype(int)
    expected = df.drop('rings', axis=1)

    # Test the result
    assert output.equals(expected)
    assert type(output) is pd.DataFrame



def test_split_data():
    output = md.split_data(initial_df_expected)
    assert type(output) is dict

    assert output["X_train"].equals(data_splitted_expected["X_train"])
    assert type(output["X_train"]) is pd.DataFrame
    assert not output["X_train"].isnull().values.any()

    assert output["X_test"].equals(data_splitted_expected["X_test"])
    assert type(output["X_test"]) is pd.DataFrame
    assert not output["X_test"].isnull().values.any()

    assert np.array_equal(output["y_train"], data_splitted_expected["y_train"])
    assert type(output["y_train"]) is np.ndarray
    assert not np.isnan(output["y_train"]).any()

    assert np.array_equal(output["y_test"], data_splitted_expected["y_test"])
    assert type(output["y_test"]) is np.ndarray
    assert not np.isnan(output["y_test"]).any()



def test_preprocessing_data():
    output =  md.preprocessing_data(data_splitted_expected)
    feature_sex = ["sex__M", "sex__F", "sex__I"]
    # Test the result
    assert type(output) is dict

    assert output["X_train"].equals(data_preprocessed_expected["X_train"])
    assert type(output["X_train"]) is pd.DataFrame
    assert not output["X_train"].isnull().values.any()
    assert all(i in list(output["X_train"].columns) for i in feature_sex)

    assert output["X_test"].equals(data_preprocessed_expected["X_test"])
    assert type(output["X_test"]) is pd.DataFrame
    assert not output["X_test"].isnull().values.any()
    assert all(i in list(output["X_test"].columns) for i in feature_sex)

    assert np.array_equal(output["y_train"], data_preprocessed_expected["y_train"])
    assert type(output["y_train"]) is np.ndarray
    assert not np.isnan(output["y_train"]).any()

    assert np.array_equal(output["y_test"], data_preprocessed_expected["y_test"])
    assert type(output["y_test"]) is np.ndarray
    assert not np.isnan(output["y_test"]).any()


def test_train_model():
    output_model =  md.train_model(data_preprocessed_expected)
    model_expected

    # data pre-processed
    my_abalone = pd.DataFrame({'sex__M': [0],
                'sex__F': [0],
                'sex__I': [1],
                'length__norm': [0.608],
                'diameter__norm': [0.546],
                'height__norm': [0.119],
                'whole_weight__norm': [0.217],
                'shucked_weight__norm': [0.175],
                'viscera_weight__norm': [0.209],
                'shell_weight__norm': [0.173]})


    output_prediction = output_model.predict(my_abalone)
    expected_prediction = model_expected.predict(my_abalone)

    assert type(output_model) is RandomForestClassifier
    assert output_prediction == expected_prediction



def test_predict_abalone():

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

    output_prediction = md.predict_abalone(my_abalone, model_expected)

    assert type(output_prediction) is dict
    assert "label" in output_prediction
    assert "probability" in output_prediction
    assert output_prediction == prediction_expected



test_load_data_abalone()
test_split_data()
test_preprocessing_data()
test_train_model()
test_predict_abalone()