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

USE_SHORT_PARAMS = True # use short params, if USE_SHORT_PARAMS= False, it  supports getting past 3 years data only, the training accuracy is not good.
TRAINING_START_DATE = '1995-01-01'
PREDICT_START_DATE = '2018-06-01'
TARGET_COLUMN = 'close'
WINDOW = 20