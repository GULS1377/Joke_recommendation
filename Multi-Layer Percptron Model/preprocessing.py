import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split


def data_preprocess(datapath):
    """
        Load and preprocess data
    """

    # load data
    filename = os.path.join(datapath, 'jester_ratings.dat')
    rating_columns = ['uid', 'jid', 'rating']
    ratings = pd.read_csv(filename, sep='\t\t', header=None, names=rating_columns, usecols=[0, 1, 2], engine='python')

    ratings = ratings[ratings['uid'] <= 5000]  # keep the ratings where uid <= 5000

    frame = ratings
    x = frame.drop(columns='rating', axis=1)  # axis=1 --> to drop columns, but not rows
    y = frame['rating']

    # split data to train set and test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # print(x_train)
    # print(x_test)
    # print(y_train)
    # print(y_test)
    return x_train, y_train, x_test, y_test
