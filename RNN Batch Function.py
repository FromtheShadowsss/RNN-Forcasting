# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:58:39 2022

@author: Yilun Li, Zhehao Fan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.layers import GRU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN,LSTM
from keras.layers import Dropout

import os

def Data_collection(file):
    
    #Data collection and formatting
    
    #read file

    path = "./" + file
    filenames = os.listdir(path)
    
    #create namespace of ETFs
    name_data = pd.read_csv('./factor_data/bolling_down_120.csv', index_col = 0)
    namespace = name_data.columns
    
    #create namesapce of factors
    factor_names = []
    for i in filenames:   
        factor_names.append(i[:-4])
    
    #read all factor data
    for i in factor_names:
        globals()["factors_" + i] = pd.read_csv("./factor_data/" + i + ".csv", index_col = 0)

    factor_names.remove('etf_ret')

    #create new table for each ETF, and append factor data
    for j in namespace:
        globals()["df_"+ j] = pd.DataFrame(factors_etf_ret[j])
        globals()["df_"+ j].columns = ['Return']
        
        for k in factor_names:
            df_temp = pd.DataFrame(globals()["factors_" + k][j])
            df_temp.columns = [k]
            globals()["df_"+ j] = pd.concat([globals()["df_"+ j], df_temp], axis = 1)
        globals()['df_' + j].drop('2015-01-06', inplace = True)
        globals()['df_' + j] = globals()['df_' + j].dropna(axis = 0, how = 'any')
        
#file = 'factor_data'      
        
def RNN_learn(df, windows = 90):
    df_train = np.array(df.loc['2015-06-23':'2019-12-31', :])
    df_test = np.array(df.loc['2020-01-02':, :])
    
    #standarlization
   
    df_train = df_train.reshape(-1, 17)

    factor_names=['etf_ret',
     'bolling_down_120',
     'bolling_down_20',
     'bolling_up_120',
     'bolling_up_20',
     'kurt_120',
     'kurt_20',
     'MAC_120',
     'MAC_20',
     'mean_120',
     'mean_20',
     'Sharpe_120',
     'Sharpe_20',
     'skew_120',
     'skew_20',
     'std_120',
     'std_20']
    
    for i in range(len(factor_names)):
        if i==0:
            globals()['sc'+ factor_names[i]] = MinMaxScaler(feature_range = (0, 1))        
            globals()['df_train_scaled'+ factor_names[i]] = globals()['sc'+ factor_names[i]].fit_transform(df_train[:,i].reshape(-1, 1))
            df_train_scaled = globals()['df_train_scaled'+ factor_names[i]]
        else:
            globals()['sc'+ factor_names[i]] = MinMaxScaler(feature_range = (0, 1))        
            globals()['df_train_scaled'+ factor_names[i]] = globals()['sc'+ factor_names[i]].fit_transform(df_train[:,i].reshape(-1, 1))
            df_train_scaled = np.c_[df_train_scaled, globals()['df_train_scaled'+ factor_names[i]]]
    df_train_scaled=np.array(df_train_scaled)     
    #for i in range(len(df)):
       #df_train = df.iloc[i:i + windows, :]
    print(df_train_scaled.shape)
        
    X_train = []
    y_train = []
    for i in range(windows, len(df_train_scaled)):
        X_train.append(df_train_scaled[i-windows:i, :])
        y_train.append(df_train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    print(X_train.shape)
    print(y_train.shape)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 17))
    print(X_train.shape)

    #RNN model
    # Start sequential model
    regressor = Sequential()
    
    # outer layer and hidden layer 
    #regressor.add(SimpleRNN(units = 32, input_shape = (X_train.shape[1], 17)))
    #regressor.add(Dropout(0.5))
    #regressor.add(LSTM(units = 32, input_shape = (X_train.shape[1], 17),return_sequences=True))
    regressor.add(LSTM(units = 32, input_shape = (X_train.shape[1], 17)))

    #regressor.add(SimpleRNN(units = 64, input_shape = (X_train.shape[1], 17)))
    regressor.add(Dropout(0.2))

    #regressor.add(SimpleRNN(16,return_sequences=False))
    #regressor.add(Dropout(0.2))

    # linear outer layers
    regressor.add(Dense(units = 1))
    
    # optimization algorithm adam， target function MSE
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])
    
    # model training
    history = regressor.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split=0.1)
    
    regressor.summary()
    
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    loss = history.history['loss']

    epochs = range(1, len(acc) + 1)
     
    plt.title('Loss')
    plt.plot(epochs, val_loss, 'blue', label='Validation loss')
    plt.plot(epochs, loss, 'Red', label='Training loss')

    plt.legend()
    plt.show()
    regressor.summary()
    
    # test data   
    real_return = np.array(df_test[:,0].reshape(-1,1))
    
    dataset_total = df
    inputs = dataset_total[len(dataset_total) - len(df_test) - windows:].values
    inputs = inputs.reshape(-1,17)
    #inputs = sc.transform(inputs)
    for i in range(len(factor_names)):
        if i==0:
            globals()['inputs'+ factor_names[i]] = globals()['sc'+ factor_names[i]].fit_transform(inputs[:,i].reshape(-1, 1))
            inputs2 = globals()['inputs'+ factor_names[i]]     
        else:            
            globals()['inputs'+ factor_names[i]] = globals()['sc'+ factor_names[i]].fit_transform(inputs[:,i].reshape(-1, 1))
            inputs2 = np.c_[inputs2, globals()['inputs'+ factor_names[i]]]
    inputs2=np.array(inputs2)
    
 

    # obtain test set
    X_test = []
    for i in range(windows, len(inputs2)):
        X_test.append(inputs2[i-windows:i])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 17))
    
    
    # Model prediction
    predicted_return = regressor.predict(X_test)
    # Inverse normalization
    predicted_return = scetf_ret.inverse_transform(predicted_return)
    # Model evaluation
    print('MSE',sum(pow((predicted_return - real_return),2))/predicted_return.shape[0])
    print('MAE',sum(abs(predicted_return - real_return))/predicted_return.shape[0])

    # Difference between prediction and real data; visualization
    plt.plot(real_return, color = 'red', label = 'Real Return')
    plt.plot(predicted_return, color = 'blue', label = 'Predicted Return')
    plt.title('Return Prediction')
    plt.xlabel('samples')
    plt.ylabel('Return')
    plt.legend()
    plt.show()
    
    return predicted_return

Data_collection('factor_data')
RNN_learn(df_CDC, windows = 90)

    
#create namespace of ETFs
name_data = pd.read_csv('./factor_data/bolling_down_120.csv', index_col = 0)
namespace = name_data.columns
for i in range(len(namespace)):
    if i == 0:
        
        ar = RNN_learn(globals()['df_' + namespace[i]])
    else:
        ar = np.c_[ar, RNN_learn(globals()['df_' + namespace[i]])]
                
ar = np.DataFrame(ar)
ar.columns = namespace
