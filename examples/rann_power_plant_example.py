###############################################################################
# Author: Bilgehan Kösem
# E-mail: bilkos92@gmail.com
# Date created: 26.19.2020
# Date last modified: 26.19.2020
# Python Version: 3.8
###############################################################################

###############################################################################
# References:
# 1) Link to Combined Cycle Power Plant Data Set:
# https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rann import RANN
from math import sqrt
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    df = pd.read_excel('ccpp.xlsx')
    
    x = df.loc[:, df.columns != 'PE']
    y = df['PE']

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    df_values = sc.fit_transform(df.values)
   
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df_values[:,:-1],
                                                        df_values[:,-1], 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    
    train_y = train_y.reshape(-1,1)
    test_y = test_y.reshape(-1,1)
    
    rann = RANN([train_x.shape[1],10,1],
                weight_scaler = 0.05,
                loss_function='mse',
                learning_rate = 0.01,
                optimizer='adam')
    
    batch_error_list = []
    error_list = []
    batch_number = 10
    for i in range(50):
        loss=0
        for j in range(len(train_x)):
            rann.calculate_rate()
            output = rann.feedforward(train_x[j])

            if rann.loss_func == 'mse':
                loss += float(0.5*sum([(output[o]-train_y[j][o])**2 for o in range(len(train_y[j]))]))
                d_L_d_y = output - train_y[j]
            
            if j % batch_number != (batch_number-1):
                pass
            else:
                loss += rann.backpropagation(train_y[j],d_L_d_y)
                error = float(loss/batch_number)
                batch_error_list.append(error)
                loss=0

        batch_mean = np.mean(batch_error_list)
        error_list.append(batch_mean)
        batch_error_list.clear()
        print('Epoch: '+str(i)+' MSE: '+ str(batch_mean))
        
    y_prediction_test=[]
    y_prediction_train=[]
    for j in range(len(test_x)):
        y_prediction_test.append(float(rann.feedforward(test_x[j])))
    for j in range(len(train_x)):
        y_prediction_train.append(float(rann.feedforward(train_x[j])))

    y_prediction_test = np.atleast_2d(y_prediction_test).reshape(-1,1)
    y_prediction_train = np.atleast_2d(y_prediction_train).reshape(-1,1)
    
    rmse_train = sqrt(mean_squared_error(train_y, y_prediction_train))
    rmse_test = sqrt(mean_squared_error(test_y, y_prediction_test))
    
    print("Test Set RMS Error:",str('%.3f'%rmse_test))
    print("Training Set RMS Error:",str('%.3f'%rmse_train))
    
    plt.plot(error_list);plt.xlabel("Epoch");plt.ylabel("RMS Error");plt.title("RANN Training Loss")
        