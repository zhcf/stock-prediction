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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.recurrent import LSTM
from keras.metrics import binary_accuracy
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import os
from const import *


g_data_train_directory = DIR_DATA_TRAIN_FULL_PARAMS
g_model_directory = DIR_MODEL_FULL_PARAMS
g_data_predict_directory = DIR_DATA_PREDICT_FULL_PARAMS

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

    y_train = all[:, -1][:, -1] - all[:, -2][:, 0]
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
    all = all[-3:-1]

    return all

def build_model(model_input_dim, model_window):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=model_input_dim,
        input_length=model_window,
        output_dim=20,
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=1))
    model.add(Activation("sigmoid"))

    model.compile(loss="mse", optimizer="rmsprop", metrics=[binary_accuracy])
    return model

def train(stock_index):
    try:
        df = pd.read_csv(g_data_train_directory + stock_index)
        x_train, y_train = preprocess_training_data(df, 20)
        model = build_model(x_train.shape[2], 20)
        print("start stock_index:%s" % (stock_index))
        hyper_parameters = [(4,64)]
        for hyper_parameter in hyper_parameters:
            from keras.callbacks import ModelCheckpoint
            checkpoint = ModelCheckpoint(g_model_directory+stock_index+".model", monitor='binary_accuracy',
                                         verbose=1, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]
            model.fit(
                x_train,
                y_train,
                batch_size=hyper_parameter[0],
                nb_epoch=hyper_parameter[1],
                validation_split=0.1,
                callbacks = callbacks_list,
                verbose=0)
            # print("%s: Shape test_x %s, test_y %s" % (stock_index, str(X_test.shape), str(y_test.shape)))
            score = model.evaluate(x_train, y_train, verbose=1)
            result = "stock_index:%s,hyper_parameter:[%s,%s], binary_accuracy:%f"%(
                stock_index,hyper_parameter[0],hyper_parameter[1],score[1])
            print(result)
            with open(g_model_directory+stock_index+".readme", "w") as f:
                f.write(result)
    except Exception as e:
        print(e)

def predict(stock_index):
    from keras.models import load_model
    model = load_model(g_model_directory + stock_index + ".model")
    df = pd.read_csv(g_data_predict_directory + stock_index)
    data = preprocess_infernece_data(df, 20)
    pred = model.predict(data)
    print(stock_index)
    print(pred)

if __name__ == "__main__":
    # stocks = get_stocks()
    # for (stock_index, stock_name) in stocks.items():
    #     train(stock_index)
    stock_index = "000063"
    # train(stock_index)
    predict(stock_index)


    # train()
    # predict()




