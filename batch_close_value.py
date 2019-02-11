from data_preparation import get_stocks_data
from const import *
from stock_prediction_close_value import predict, train
import sys

if __name__ == '__main__':
    stocks = get_stocks()
    argv = sys.argv
    action = "predict"
    if len(argv) > 1:
        action = argv[1]
    if action == "predict":
        print("===================predict====================")
        get_stocks_data(stocks, get_predict_data=True)
        predict(stocks)
    elif action == "train":
        print("===================train====================")
        get_stocks_data(stocks, get_predict_data=False)
        train(stocks)

