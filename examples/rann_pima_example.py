###############################################################################
# Author: Bilgehan Kösem
# E-mail: bilkos92@gmail.com
# Date created: 26.19.2020
# Date last modified: 26.19.2020
# Python Version: 3.8
###############################################################################

###############################################################################
# References:
# 1) Link to PIMA Indians Diabetes Data Set:
# https://www.kaggle.com/uciml/pima-indians-diabetes-database
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from rann import RANN
from numpy import loadtxt

if __name__ == '__main__':

    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    X = dataset[:,0:8]
    y = dataset[:,8]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(-1, 1))
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    train_x = X_train#.reshape(-1,1)
    train_y = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    rann = RANN([X_train.shape[1],10,1],
                weight_scaler = 0.05,
                loss_function='mse',
                learning_rate = 0.01,
                optimizer='adam')
    
    batch_error_list = []
    error_list = []
    batch_number = 10
    for i in range(100):
        loss=0
        num_correct = 0 #Counter for right prediction for each epoch
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

    ###########################################################################
    #1x1 output
    if rann.shape[2] == 1:
        y_prediction_test=[]
        y_prediction_train=[]
        for j in range(len(X_test)):
            y_prediction_test.append(float(rann.feedforward(X_test[j])))
        for j in range(len(train_x)):
            y_prediction_train.append(float(rann.feedforward(train_x[j])))
    
        y_pred_train = (np.array(y_prediction_train) > 0.5)
        cm_train = confusion_matrix(y_pred_train, train_y)
        cm_train_acc = (cm_train[0,0]+cm_train[1,1])/len(y_pred_train)*100
        
        y_pred_test = (np.array(y_prediction_test) > 0.5)
        cm_test = confusion_matrix(y_pred_test, y_test)   
        cm_test_acc = (cm_test[0,0]+cm_test[1,1])/len(y_pred_test)*100
        
        print("Test Accuracy:",str('%.2f'%cm_test_acc))
        print("Train Accuracy:",str('%.2f'%cm_train_acc))
       
    ###########################################################################
    #2x1 output
    if rann.shape[2] == 2:
        
        y_prediction_test=[]
        y_prediction_train=[]
        for j in range(len(X_test)):
            y_prediction_test.append(np.argmax(rann.softmax(rann.feedforward(X_test[j]))))
        for j in range(len(train_x)):
            y_prediction_train.append(np.argmax(rann.softmax(rann.feedforward(train_x[j]))))
       
        cm_train = confusion_matrix(y_prediction_train, train_y)
        cm_train_acc = (cm_train[0,0]+cm_train[1,1])/len(y_prediction_train)*100
        
        cm_test = confusion_matrix(y_prediction_test, y_test)   
        cm_test_acc = (cm_test[0,0]+cm_test[1,1])/len(y_prediction_test)*100

    plt.plot(error_list);plt.xlabel("Epoch");plt.ylabel("RMS Error");plt.title("RANN Training Loss")
