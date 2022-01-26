#!/usr/bin/env python
# coding: utf-8

"""
    Binary classification of the abalones to predict their age.
    (0:less than 15 years, 1:equal/more)

    Dataset
    =============
    "../data/abalone.data", 4177 observations, 9 features.

    ML Model used
    =============
    Random Forest Classification (bad precision)

    Context
    =============
    Predicting the age of abalone from physical measurements.  The age of
    abalone is determined by cutting the shell through the cone, staining it,
    and counting the number of rings through a microscope -- a boring and
    time-consuming task.  Other measurements, which are easier to obtain, are
    used to predict the age.

    Features
    ============
        - sex / nominal / -- / M, F, and I (infant)
        - Length / continuous / mm / Longest shell measurement
        - Diameter / continuous / mm / perpendicular to length
        - Height / continuous / mm / with meat in shell
        - Whole weight / continuous / grams / whole abalone
        - Shucked weight / continuous / grams / weight of meat
        - Viscera weight / continuous / grams / gut weight (after bleeding)
        - Shell weight / continuous / grams / after being dried
        - Rings / integer / -- / +1.5 gives the age in years

    Target feature : ring of the abalone
    ====================================
    'target' (based on 'rings' feature) :
        - 0 : less than 10 rings ( < 15 years)
        - 1 : more than 10 rings ( >= 15 years)

"""
import pickle as pkl

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data_abalone(path):
    """ Load abalone data into a panda dataframe then transform the feature
        'rings' (int) into 'target' (int).

        :param str path: Path of the abalone dataset.

        :returns: df: The dataset of the experience with
                       feature 'rings' transformed into 'target' feature.
        :rtype: pandas.DataFrame
    """
    names = ['sex', 'length', 'diameter', 'height', 'whole_weight',
             'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    df = pd.read_csv(path, header=None, names=names)

    df['target'] = (df['rings'] >= 10).astype(int)
    df = df.drop('rings', axis=1)
    return df


def split_data(df, test_size=0.33, seed=42):
    """ Split the dataframe into test and train data.

        :param pandas.DataFrame df: The dataset of the experience.
        :param  float/int test_size: Proportion of the dataset to
                    include to the test set.
        :param int seed: Value to have reproducible results.

        :returns: data : Dictionary structured as:
            {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}

        :rtype: dict

    """
    # Separating target from features
    y = np.array(df['target'])
    X = df.drop('target', axis=1)

    # Shuffling and splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed)

    data = {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}

    return data


def dummy_encode(in_df):
    """ Encoding categorical features; here 'sex' of the abalone.
    Male (1,0,0), Female(0,1,0) and Infant(0,0,1).

    :param pandas.DataFrame in_df: X_train (or X_test).
    :returns: out_df: X_train (or X_test) processed.
    :rtype: pandas.DataFrame

    """
    out_df = in_df.copy()
    cat_features = {'sex': ['M', 'F', 'I']}

    for feature, values in cat_features.items():
        for value in values:
            dummy_name = '{}__{}'.format(feature, value)
            out_df[dummy_name] = (out_df[feature] == value).astype(int)

        del out_df[feature]

    return out_df


def minmax_scale(in_df):
    """ Rescaling numerical features based on their boundaries.

        :param pandas.DataFrame in_df: X_train (or X_test).
        :returns: out_df: X_train (or X_test) processed.
        :rtype: pandas.DataFrame
    """
    out_df = in_df.copy()

    # Numeric Features (Min-max scaling)
    boundaries = {
        'length': (0.075000, 0.815000),
        'diameter': (0.055000, 0.650000),
        'height': (0.000000, 1.130000),
        'whole_weight': (0.002000, 2.825500),
        'shucked_weight': (0.001000, 1.488000),
        'viscera_weight': (0.000500, 0.760000),
        'shell_weight': (0.001500, 1.005000)
    }

    for feature, (min_val, max_val) in boundaries.items():
        col_name = '{}__norm'.format(feature)

        out_df[col_name] = round(
            (out_df[feature] - min_val) / (max_val - min_val), 3)
        out_df.loc[out_df[col_name] < 0, col_name] = 0
        out_df.loc[out_df[col_name] > 1, col_name] = 1

        del out_df[feature]

    return out_df


def preprocessing_data(data_in):
    """ Get 'X_train' and 'X_test' from 'data_in'(dict), apply one-hot encoding
        and rescaling on them then return the dict with X_train and X_test
        processed.

        :param dict data_in: Dictionary structured as:
                        {"X_train": X_train, "X_test": X_test,
                        "y_train": y_train, "y_test": y_test}
        :returns: data_out: The same dictionary on the input with X_train
                        and X_test pre-processed.
        :rtype: dict
    """
    data_out = data_in.copy()
    # Categorical Features : One-hot encoding
    X_train = dummy_encode(data_out["X_train"])
    X_test = dummy_encode(data_out["X_test"])

    # Numerical Features : Rescaling
    X_train = minmax_scale(X_train)
    X_test = minmax_scale(X_test)

    data_out["X_train"] = X_train
    data_out["X_test"] = X_test

    return data_out

def train_model(data):
    """Train and return the model.

   :param dict data: Dictionary structured as:
                    {"X_train": X_train, "X_test": X_test,
                    "y_train": y_train, "y_test": y_test}

   :returns: model: Random Forest model
        trained.
   :rtype: sklearn.RandomForestClassifier

    """
    clf = RandomForestClassifier(
        n_estimators=100,  # number of trees
        n_jobs=-1,  # parallelization
        random_state=1337,  # random seed
        max_depth=10,  # maximum tree depth
        min_samples_leaf=10
    )

    model = clf.fit(data["X_train"], data["y_train"])
    return model


def print_roc_auc(model, data):
    """Print the ML ROC AUC score

       :param sklearn.RandomForestClassifier model:  ML model
       :param dict data:
            Dictionary structured as : {"X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}
    """
    train_auc = sk.metrics.roc_auc_score(data["y_train"],
                                         model.predict(data["X_train"]))
    test_auc = sk.metrics.roc_auc_score(data["y_test"],
                                        model.predict(data["X_test"]))

    print('Training ROC AUC:\t', round(train_auc, 3))
    print('Test ROC AUC:\t\t', round(test_auc, 3))


def store_model(model, path):
    """Store the ML model store at the path

       :param sklearn model: ML model to store
       :param str path: Path of the model stored
    """
    pkl.dump(model, open(path, 'wb'))


def load_model(path):
    """Load the ML model store at the path

       :param str path: Path of the model stored
    """
    model_ = pkl.load(open(path, 'rb'))
    return model_


def predict_abalone(abalone, model):
    """Return the label and the probability for one abalone.

       :param dict abalone:
            dictionary such as '{"val1":val1, "val2":val2}'.
       :param sklearn model: sklearn model.
       :returns: abalone_predict: Dictionary where "label" and "proba" are
                the keys. {"label": label, "proba": proba}
       :rtype: dict

    """
    # To make the pre-processing, a pandas.DataFrame is needed
    x_new = pd.DataFrame([abalone])

    # Pre-processing
    x_new = dummy_encode(x_new)
    x_new = minmax_scale(x_new)

    # Prediction
    label = model.predict(x_new)[0]
    proba = model.predict_proba(x_new)[0][1]

    abalone_predict = {"label": label, "probability": round(proba, 3)}
    return abalone_predict


if __name__ == "__main__":
    # 1. Data
    my_df = load_data_abalone('../data/abalone.data')

    # 2. Feature Preparation
    my_data = split_data(my_df)
    my_data = preprocessing_data(my_data)

    # 3. Training
    my_model = train_model(my_data)

    # 4. Evaluation
    print_roc_auc(my_model, my_data)

    # 5. Storing Model
    store_model(my_model, "../data/pickles/model_RF_abalone_v1.pkl")

    # 6. Example
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

    print(predict_abalone(my_abalone, my_model).values())
