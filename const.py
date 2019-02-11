#!/usr/bin/env python
# coding: utf-8

DIR_DATA_TRAIN_FULL_PARAMS = './data_train_full_params/'
DIR_DATA_TRAIN_SHORT_PARAMS = './data_train_short_params/'
DIR_DATA_PREDICT_FULL_PARAMS = './data_predict_full_params/'
DIR_DATA_PREDICT_SHORT_PARAMS = './data_predict_short_params/'

DIR_MODEL_FULL_PARAMS = './model_full_params/'
DIR_MODEL_SHORT_PARAMS = './model_short_params/'
DIR_PREDICT_FULL_PARAMS = './predict_full_params/'
DIR_PREDICT_SHORT_PARAMS = './predict_short_params/'

'''
if USE_SHORT_PARAMS= True, the record columns will be the following, it supports getting all data, the training accuracy is good 
date
交易日期(index)
open
开盘价
high
最高价
close
收盘价
low
最低价
volume
成交量
amount
成交金额
if USE_SHORT_PARAMS= False, the record columns will be the following, it supports getting past 3 years data only, the training accuracy is not good.
属性:日期 ，开盘价， 最高价， 收盘价， 最低价， 成交量， 价格变动 ，涨跌幅，5日均价，10日均价，20日均价，5日均量，10日均量，20日均量，换手率
'''
USE_SHORT_PARAMS = True
TRAINING_START_DATE = '1995-01-01'
PREDICT_START_DATE = '2018-06-01'
TARGET_COLUMN = 'close'
WINDOW = 20
THRESHOLD = 0.5



def get_stocks():
    stocks = {}
    with open("stocklist.txt", "r") as ff:
        lines = ff.readlines()
        for line in lines:
            items = line.split(",")
            if len(items[0]) > 0:
                stocks[items[0]] = items[1]
    return stocks