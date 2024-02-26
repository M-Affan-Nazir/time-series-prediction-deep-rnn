import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf



def predicting():
    rnn = tf.keras.models.load_model("./trainedRnn")
    print(rnn.summary())
    dataset_train = pd.read_csv("./google-stock-history/stock_train.csv")   
    train = dataset_train.iloc[:,1:2]
    train = np.array(train)

    dataset_test = pd.read_csv("./google-stock-history/stock_test.csv")
    testStock = dataset_test.iloc[:,1:2]
    testStock = np.array(testStock)

    totalData = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
    #axis = 0 ; will adds values of dataset_test['Open'] underneath the values of dataset_train['Open']! Basically vertical extension. Array extend kardaita.
    #axis = 1 ; will add a new column! values of dataset['Open'] in the new column! But since datase+train['Open'] has like 1198 values, but datset_test['Open'] has only 20; 1178 (= 1198-20) values in the second column will automatically get the Null/NaN value; kyonkay 2nd column has dataset_train['Open'] values, and dataset_train['Open'] only has 20. Baki 1178 will be null.
    #dataset_test and dataset_train is a dictionary. Agay, dataset_train['Open'] is a dictionary as well. to concatinate dictionaries, we use the pd.concat() function. Otherwise agar list hoti (using np.array(dataset_train['Open'], phir tou normally bhi concat kar saktay thay, like total = array1 + array2))
    
    #Now, basically Taking 60 stocks (October, November, December 2016; from train)  + testing stocks (January):
    inputs = totalData[len(totalData) - len(dataset_test) - 60:]
    inputs = np.array(inputs)
    # len(totalData)-len(dataset_test) = total stock entries in training set
    # len(totalData)-len(dataset_test) - 60 = training set - 60
    # [len(totalData) - len(dataset_test) - 60:] = take 60 stock from training set + all the test stocks!
    # len(inputs) = 80 (60 from training, all 20 from test)  
 
    inputs = inputs.reshape(-1,1)
    #reshape(-1,1) = -1 tells to " calculate the number of rows required to maintain the same number of elements in the reshaped array".  1 tells to have only 1 column
    #So, now inputs = a number-ofelementsx1 vector! and each row is an array containing only 1 stock price
    ''' [1,2,3] -> [ [1],
                     [2],
                     [3] ]

    '''

    normalize = MinMaxScaler(feature_range=(0,1))
    normInput = normalize.fit_transform(inputs)
    test = []
    #array, wher each element is an array of 60 stock prices; so we can predict the next stock price.
    #1st 60 = all from train; so we predict 1st Jan price
    #2nd 60 = 59 from train + 1st january (from test, actual price, not predicted), so we predict 2nd January price
    #in this manner, we get predict prices from 1st Jan to 20 Jan (20 values, since 20 financial days, beech mai chut bhi)
    #We can then compare predicted prices with actual Jan prices (test data)!
    for i in range(60, len(normInput)):
        test.append(normInput[i-60:i])
    

    test = np.array(test)
    #Reshaping 'test' to three dimensional array!
    test = np.reshape(test, (test.shape[0],test.shape[1],1))
    



    #Predicting:
    predictions = rnn.predict(test)
    #Reverse Normalization, Get actual price:
    predictionPrice = normalize.inverse_transform(predictions)

    visualize(testStock,predictionPrice)


def visualize(actual,predicted):
    plt.plot(actual, color='red', label="Actual Stock Price")
    plt.plot(predicted, color='blue',label="Predicted Stock Price by Model")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


predicting()