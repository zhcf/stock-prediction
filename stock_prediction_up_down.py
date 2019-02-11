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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep
import os
from const import *

if not os.path.exists(DIR_MODEL_FULL_PARAMS):
    os.mkdir(DIR_MODEL_FULL_PARAMS)
if not os.path.exists(DIR_MODEL_SHORT_PARAMS):
    os.mkdir(DIR_MODEL_SHORT_PARAMS)

if not os.path.exists(DIR_PREDICT_FULL_PARAMS):
    os.mkdir(DIR_PREDICT_FULL_PARAMS)
if not os.path.exists(DIR_PREDICT_SHORT_PARAMS):
    os.mkdir(DIR_PREDICT_SHORT_PARAMS)

if USE_SHORT_PARAMS:
    g_data_predict_directory = DIR_DATA_PREDICT_SHORT_PARAMS
    g_data_train_directory = DIR_DATA_TRAIN_SHORT_PARAMS
    g_model_directory = DIR_MODEL_SHORT_PARAMS
    g_predict_directory = DIR_PREDICT_SHORT_PARAMS
else:
    g_data_predict_directory = DIR_DATA_PREDICT_FULL_PARAMS
    g_data_train_directory = DIR_DATA_TRAIN_FULL_PARAMS
    g_model_directory = DIR_MODEL_FULL_PARAMS
    g_predict_directory = DIR_PREDICT_FULL_PARAMS


def preprocess_training_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.values

    sequence_length = seq_len + 1
    all = []
    for index in range(len(data) - sequence_length):
        all.append(data[index: index + sequence_length])

    all = np.array(all)
    row = round(0.9 * all.shape[0])

    samples = all[:, : -1]
    x_train = samples[: int(row), :]
    x_test = samples[int(row):, :]

    samples = samples.reshape(samples.shape[0], seq_len * amount_of_features)
    x_train = x_train.reshape(x_train.shape[0], seq_len * amount_of_features)
    x_test = x_test.reshape(x_test.shape[0], seq_len * amount_of_features)
    preprocessor_sample = prep.StandardScaler().fit(samples)
    x_train = preprocessor_sample.transform(x_train)
    x_test = preprocessor_sample.transform(x_test)
    x_train = x_train.reshape(x_train.shape[0], seq_len, amount_of_features)
    x_test = x_test.reshape(x_test.shape[0], seq_len, amount_of_features)

    all[:, -1, -1] = all[:, -1, -1] - all[:, -2, -1]
    all[:, -1, -1][all[:, -1, -1] > 0] = 1
    all[:, -1, -1][all[:, -1, -1] <= 0] = 0
    y = all[:, -1][:, -1]
    y_train = y[: int(row)]
    y_test = y[int(row):]
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    return [x_train, y_train, x_test, y_test]


def preprocess_inference_data(stock, seq_len):
    date = stock.values[:, 0][seq_len:]
    date = np.append(date, (0))
    y1 = stock.values[:, -1][seq_len-1:-1]
    y2 = stock.values[:, -1][seq_len:]
    y = y2 -y1
    y[:][y[:]>0] = 1
    y[:][y[:]<=0] = 0
    y = np.append(y, (0))

    col_list = stock.columns.tolist()
    col_list.remove('date')
    stock = stock[col_list]
    amount_of_features = len(stock.columns)
    data = stock.values

    all = []
    for index in range(len(data) - seq_len + 1):
        all.append(data[index: index + seq_len])

    all = np.array(all)

    samples = all[:, :, :]
    samples = samples.reshape(samples.shape[0], seq_len * amount_of_features)
    preprocessor_sample = prep.StandardScaler().fit(samples)
    samples = preprocessor_sample.transform(samples)
    samples = samples.reshape(samples.shape[0], seq_len, amount_of_features)

    return [date, samples, y]


# ## Build the LSTM Network
#
# Here we will build a simple RNN with 2 LSTM layers.
# The architecture is:
#
#     LSTM --> Dropout --> LSTM --> Dropout --> Fully-Conneted(Dense)

# In[5]:

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
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    return model


def build_model_1(model_input_dim, model_window):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=model_input_dim,
        input_length=model_window,
        output_dim=20,
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        20,
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        100,
        return_sequences=False))
    model.add(Dropout(0.3))
    # model.add(Flatten())

    model.add(Dense(
        output_dim=1))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    return model


def train(stocks):
    stocks_accuracy = []
    for (stock_index, stock_name) in stocks.items():
        try:
            df = pd.read_csv(g_data_train_directory + stock_index)
            col_list = df.columns.tolist()
            col_list.remove('date')
            df = df[col_list]
            X_train, y_train, X_test, y_test = preprocess_training_data(df, WINDOW)
            model = build_model(X_train.shape[2], WINDOW)

            # hyper_parameters = [(256, 1),(256,5)]
            hyper_parameters = [(64, 10), (128, 10), (64, 50), (128, 50), (256, 50),
                                (64, 100), (128, 100), (128, 300), (256, 300),
                                (512, 300), (128, 500), (256, 500), (512, 500)]
            min_error = 10000.0
            selected_hyper_parameter = None
            for hyper_parameter in hyper_parameters:
                model.fit(
                    X_train,
                    y_train,
                    batch_size=hyper_parameter[0],
                    nb_epoch=hyper_parameter[1],
                    validation_split=0.0,
                    verbose=0)
                score = model.evaluate(X_test, y_test, verbose=0)
                error = math.sqrt(score[0])
                if error < min_error:
                    selected_hyper_parameter = hyper_parameter
                    model.save(g_model_directory + stock_index, True)
                    min_error = error
            print(stock_index + " selected_hyper_parameter: " + str(selected_hyper_parameter))
            print(stock_index + " min_error: " + str(min_error))

            pred = model.predict(X_test)
            pred = pred.reshape(pred.shape[0])
            pred[:][pred[:] > THRESHOLD] = 1
            pred[:][pred[:] <= THRESHOLD] = 0
            y_test = y_test.reshape(y_test.shape[0])
            correct_count = 0
            for u in range(len(pred)):
                if pred[u] == y_test[u]:
                    correct_count += 1
            accuracy = float(correct_count) / float(len(pred)) * 100
            stocks_accuracy.append((stock_index, min_error, accuracy, selected_hyper_parameter[0], selected_hyper_parameter[1]))

        except Exception as e:
            print(e)

    print(stocks_accuracy)
    import csv
    with open(g_model_directory + "0000000_all_stock_result", 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(stocks_accuracy)


def predict(stocks):
    stocks_accuracy = []
    for (stock_index, stock_name) in stocks.items():
        try:
            from keras.models import load_model
            model = load_model(g_model_directory + stock_index)

            df = pd.read_csv(g_data_predict_directory + stock_index)
            date, samples, y = preprocess_inference_data(df, WINDOW)

            pred = model.predict(samples)
            pred = pred.reshape(pred.shape[0])
            pred[:][pred[:] > THRESHOLD] = 1
            pred[:][pred[:] <= THRESHOLD] = 0
            y[-1] = pred[-1]
            a = [(y[-5], pred[-5]), (y[-4], pred[-4]), (y[-3], pred[-3]), (y[-2], pred[-2]), (y[-1], pred[-1])]
            print(stock_index + str(a))


            correct_count = 0
            for u in range(len(pred)):
                if pred[u] == y[u]:
                    correct_count += 1
            accuracy = float(correct_count) / float(len(pred)) * 100
            stocks_accuracy.append((stock_index, accuracy))

            import csv
            with open(g_predict_directory + stock_index, 'w', newline='') as f:
                csvwriter = csv.writer(f)
                rows = []
                rows.append(("date", "actual", "pred"))
                for i in range(len(pred)):
                    rows.append((date[i], y[i], pred[i]))
                csvwriter.writerows(rows)

        except Exception as e:
            print(e)

    print(stocks_accuracy)
    import csv
    with open(g_predict_directory + "0000000_all_stock_result", 'w', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(stocks_accuracy)


if __name__ == "__main__":
    stocks = get_stocks()
    train(stocks)
    # predict(stocks)
    # USE_SHORT_PARAMS = False
    # if USE_SHORT_PARAMS:
    #     g_data_predict_directory = DIR_DATA_PREDICT_SHORT_PARAMS
    #     g_data_train_directory = DIR_DATA_TRAIN_SHORT_PARAMS
    #     g_model_directory = DIR_MODEL_SHORT_PARAMS
    #     g_predict_directory = DIR_PREDICT_SHORT_PARAMS
    # else:
    #     g_data_predict_directory = DIR_DATA_PREDICT_FULL_PARAMS
    #     g_data_train_directory = DIR_DATA_TRAIN_FULL_PARAMS
    #     g_model_directory = DIR_MODEL_FULL_PARAMS
    #     g_predict_directory = DIR_PREDICT_FULL_PARAMS
    # train()
    # predict()
    '''the below is for multi gpu running to accelerate the training, python stock_prediction_close_value.py 4 0,  python stock_prediction_close_value.py 4 1'''
    '''
    stocks = get_stocks()
    stock_count = len(stocks)
    import sys

    argv = sys.argv
    gpu_count = int(argv[1])
    gpu_index = int(argv[2])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sub_stocks = {}
    index = 0
    stocks_per_gpu = int(stock_count / gpu_count) + 1
    print(stocks_per_gpu)
    for (stock_index, stock_name) in stocks.items():
        if index >= stocks_per_gpu * gpu_index and index < stocks_per_gpu * (gpu_index + 1):
            sub_stocks[stock_index] = stock_name
        index += 1
    stocks_accuracy = train(sub_stocks)
    '''



