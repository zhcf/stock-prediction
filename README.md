# Stock Prediction with Recurrent Neural Network

This is forked from _https://github.com/Kulbear/stock-prediction_ 
 
The data we used is from Chinese stock.

## Requirements

- Python 3.5
- TuShare 0.7.4
- Pandas 0.19.2
- Keras 1.2.2
- Numpy 1.12.0
- scikit-learn 0.18.1
- TensorFlow 1.0 (GPU version recommended)

I personally recommend you to use Anaconda to build your virtual environment. And the program probably cost a significant time if you are not using the GPU version Tensorflow.


## Predict The Stock Close Value
This is used to train the model to predict the stock price.  
1. Add the stock code into the stocklist.txt  
2. python batch_close_value.py train  
3. python batch_close_value.py predict

The step 2 just need run at first time, after the model is trained, next time, you can just run step 3.

## Predict The Stock Up or Down
This is used to train the model to predict the stock rise up or go down.
1. Add the stock code into the stocklist.txt  
2. python batch_up_down.py train  
3. python batch_up_down.py predict

The step 2 just need run at first time, after the model is trained, next time, you can just run step 3.

## Demo for Predicting Stock Close Value

<div style="text-align:center">
	<img src="https://cloud.githubusercontent.com/assets/14886380/25383467/de39614e-29ee-11e7-9a3c-ac9e34720b54.png" alt="Training Result Demo" style="width: 450px;"/>
</div>

## Reference

- [Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras](http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Understanding LSTM Networks by Christopher Olah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
