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

If this project is cloned to /opt/  
docker run -it --gpus all -v /opt:/opt/ zhcf/stock:0128 /bin/bash (The zhcf/stock is the runtime ready docker image)   
pyhton stock-prediction.py  (train model or use  trained model to predict)   
python data_preparation.py (get the training data or prediction data)   

You can run the prediction tonight, the prediction result represents the probability of meansthird day's close value is higher than tomorrow's open value.
