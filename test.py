#!/usr/bin/env python
# coding: utf-8

# # Stock Prediction with Recurrent Neural Network
#
# Deep learning is involved a lot in the modern quantitive financial field. There are many different neural networks can be applied to stock price prediction problems. The recurrent neural network, to be specific, the Long Short Term Memory(LSTM) network outperforms others architecture since it can take advantage of predicting time series (or sequentially) involved result with a specific configuration.
#
# We will make a really simple LSTM with Keras to predict the stock price in the Chinese stock.

# In[1]:


import time
import math

import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import os
from const import *




def get_stocks():
    stocks = {}
    with open("stocklist.txt", "r") as ff:
        lines = ff.readlines()
        for line in lines:
            items = line.split(",")
            if len(items[0]) > 0:
                stocks[items[0]] = items[1]
    return stocks


def preprocess_training_data(df, seq_len):
    col_list = df.columns.tolist()
    col_list.remove('date')
    df = df[col_list]

    amount_of_features = len(df.columns)
    data = df.values

    sequence_length = seq_len + 1
    all = []
    for index in range(len(data) - sequence_length):
        all.append(data[index: index + sequence_length])

    all = np.array(all)
    x_train = all[:, : -1]
    x_train = x_train.reshape(x_train.shape[0], seq_len * amount_of_features)
    preprocessor_x = prep.StandardScaler().fit(x_train)
    x_train = preprocessor_x.transform(x_train)
    x_train = x_train.reshape(x_train.shape[0], seq_len, amount_of_features)

    y_train = all[:, -1][:,-1] - all[:, -2][:,0]
    y_train[y_train > 0] = 1
    y_train[y_train <= 0] = 0

    return [x_train, y_train]

def preprocess_infernece_data(df, seq_len):
    col_list = df.columns.tolist()
    col_list.remove('date')
    df = df[col_list]

    amount_of_features = len(df.columns)
    data = df.values

    sequence_length = seq_len
    all = []
    for index in range(len(data) - sequence_length):
        all.append(data[index: index + sequence_length])

    all = np.array(all)
    all = all.reshape(all.shape[0], seq_len * amount_of_features)
    preprocessor_x = prep.StandardScaler().fit(all)
    all = preprocessor_x.transform(all)
    all = all.reshape(all.shape[0], seq_len, amount_of_features)
    all = all[-2:-1]

    return all

def train():
    stocks = get_stocks()
    for (stock_index, stock_name) in stocks.items():
        try:
            df = pd.read_csv(g_data_train_directory + stock_index)
            x_train, y_train = preprocess_infernece_data(df, 20)
        except Exception as e:
            print(e)



if __name__ == "__main__":
    USE_SHORT_PARAMS = False
    g_data_train_directory = DIR_DATA_TRAIN_FULL_PARAMS
    g_model_directory = DIR_MODEL_FULL_PARAMS
    g_predict_directory = DIR_PREDICT_FULL_PARAMS
    train()

    # train()
    # predict()




