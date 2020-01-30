#!/usr/bin/env python
# coding: utf-8

# # Stock Data Preparation Demo
# 
# Today we will do a simple experiment on a stock prediction case. 
# 
# In order to save time on preparing the data, I'd like to introduce **TuShare**. You can use any other data source if you don't want to deal with Chinese stock market or you are not familiar with Chinese (TuShare's doc is all Chinese). 
# 
# In addition, you can have as many features as you want unless you keep the close price (the one we want to predict!) as the last column in your Pandas DataFrame.
# 
# **This is just a demo about how to use TuShare and deal with its data. You can find a separate (and simple) script with instructions about fetching more detailed data.**

# In[1]:


import tushare as ts # TuShare is a utility for crawling historical data of China stocks
import pandas as pd
import os
from multiprocessing import Process,Pool
from const import *

if not os.path.exists(DIR_DATA_TRAIN_FULL_PARAMS):
    os.mkdir(DIR_DATA_TRAIN_FULL_PARAMS)
if not os.path.exists(DIR_DATA_PREDICT_FULL_PARAMS):
    os.mkdir(DIR_DATA_PREDICT_FULL_PARAMS)

def get_data(stock_index, csv_name):
    try:
        print(stock_index)
        df = ts.get_hist_data(stock_index, ktype='D')
        df = df.sort_index(ascending=True)
        df = df.reset_index()
        col_list = df.columns.tolist()
        col_list.remove(TARGET_COLUMN)
        col_list.append(TARGET_COLUMN)
        df = df[col_list]
        df.to_csv(csv_name, index=False)
        validate_df = pd.read_csv(csv_name)
        validate_df.head()
    except Exception as e:
        print(e)



def get_all_stock_data(ispredictdata=True):
    stocks = {}
    with open("stocklist.txt","r") as ff:
        lines = ff.readlines()
        for line in lines:
            items = line.split(",")
            if len(items[0]) > 0:
                stocks[items[0]] = items[1]

    csv_directory = DIR_DATA_PREDICT_FULL_PARAMS
    if ispredictdata == False:
        csv_directory = DIR_DATA_TRAIN_FULL_PARAMS

    # p = Pool(3)#too many process, it will return http error 456
    for (stock_index, stock_name) in stocks.items():
        csv_name = csv_directory + stock_index
        get_data(stock_index, csv_name)
        # p.apply_async(get_data, args=(stock_index, start_date, csv_name))
    # p.close()
    # p.join()

def get_all_stock_train_data():
    get_all_stock_data(False)

def get_all_stock_predict_data():
    get_all_stock_data(True)

if __name__ == "__main__":
    get_all_stock_train_data()
    get_all_stock_predict_data()


