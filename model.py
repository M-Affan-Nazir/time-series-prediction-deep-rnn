import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

xTrain = []
yTrain = []

def getPreprocessedData():
    dataset_train = pd.read_csv("./google-stock-history/stock_train.csv")   #reads into dictionary. keys : ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']. Accessing the Volume of the 50th stock : dataset_train["Volume"][49]
    train = dataset_train.iloc[:,1:2]   #contains high of the day. Take all rows, but only column index 1 (High of the day). 1:2 --> (2 = upperbound; so excluded)
    train = np.array(train)   #turn dictionary into numpy array!

    #You can eitehr Standerdize: ( [x  - mean]  /  standard-deviation)
    #Or you can Normalize: ( [x - min]  /  (max - min) )
    normalize = MinMaxScaler(feature_range=(0,1))
    normTrain = normalize.fit_transform(train)
    return normTrain

def makeXandY():
    data = getPreprocessedData()
    global xTrain
    global yTrain
    for i in range(60,1258):
        xTrain.append(data[i-60:i,0])  
        yTrain.append(data[i,0])
        #first training entry: x = 0->59, y = 60     !(upperbound = i, but python works is that the upper bound in reality that is added is i-1! That concept, like for loop upperbound (goes up to i-1, not i). Remember?)
        #second training entry: x= 1->60, y = 61
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

def reShaping():
    global xTrain
    #adding amother dimension to the data. 
    #This new dimension will have more coresponding values (like other stock's price (which you think have an effect on the stock price you're trying to predict), Volume of stock, Open, High, Low etc etc! All those factors! )
    # xTrain currently has 2-Dimensions. 1st dimension has 1198 rows/(entries of 60-size arrays), and the 60 columns/60 stock prices)
    xTrain = np.reshape(xTrain, (xTrain.shape[0],xTrain.shape[1],1))
    #xTrain.shape[0] gives the batches of data; or the rows; i.e. 1198. xTrain.shape[1] gives the time steps/columns, which is 60. Then we change the shpe of the maytrix to 3 Dimensional, by adding 1 (as the z-coordinate)
    #Here we give 1 as the z extension, which means we have only 1 slice of 1198x60 sqaure. This is because in this example we will not be using other data values!
    #This first slice in the 3D tensor/matrix has the 'Open' values
    #If you were to add another data-value, lets say the 'Volume' of the stock; you will change the z-dimension to 2. Now you would have 2 1198x60 slices!
    #The first slice will have the 'Open' data values. The second slice will have the 'Volume' of the stock price.
    #If you were to add even another data-value, lets say the stock price of another company, you will change the z-dimension to 3. Now you will have a 3rd slice of 1198x60 available, where you can add another companies stock price!
    #The RNN will use the data in all the dimensions 'together',  to find pattern and predict future stock price!

def modelArchetecture():
    rnn = tf.keras.models.Sequential()
    rnn.add(tf.keras.layers.LSTM(units=50, return_sequences = True, input_shape = (xTrain.shape[1],1)))
    #return_sequences = True when you will add another LSTM layer infront of the current LSTM layer.
    #input_shape = (time-steps, number of 2-D slices looking from a  3-dimensional perspective)
    rnn.add(tf.keras.layers.Dropout(0.2)) #20% of neurons in the previous  hidden layer will be dropped out / ignored during Training (forward/back propagation)

    rnn.add(tf.keras.layers.LSTM(units=50, return_sequences = True))
    rnn.add(tf.keras.layers.Dropout(0.2))

    rnn.add(tf.keras.layers.LSTM(units=50, return_sequences = True))
    rnn.add(tf.keras.layers.Dropout(0.2))

    rnn.add(tf.keras.layers.LSTM(units=50))
    rnn.add(tf.keras.layers.Dropout(0.2))

    rnn.add(tf.keras.layers.Dense(units=1))


    rnn.compile(optimizer="Adam",loss="mse",metrics=["mse"])

    return rnn

def train(model):
    rnn = model
    rnn.fit(x=xTrain, y=yTrain, epochs=100, batch_size=32)




    
    

makeXandY()
reShaping()
rnn = modelArchetecture()
train(rnn)
rnn.save("./trainedRnn")