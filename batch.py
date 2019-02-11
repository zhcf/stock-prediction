from data_preparation import get_all_stock_data
from stock_prediction_close_value import predict, train
import sys

if __name__ == '__main__':
    argv = sys.argv
    action = "predict"
    if len(argv) > 1:
        action = argv[1]
    if action == "predict":
        print("===================predict====================")
        get_all_stock_data(get_predict_data=True)
        predict()
    elif action == "train":
        print("===================train====================")
        get_all_stock_data(get_predict_data=False)
        train()

