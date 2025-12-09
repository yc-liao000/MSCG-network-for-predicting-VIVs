# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2021/4/6 11:07

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.preprocessing import MinMaxScaler
import time
from random import shuffle
from torch.optim import RMSprop, Adam, LBFGS, lr_scheduler
import joblib  # save scaler
import torch.utils.data as Data
import torch.nn as nn
import torch
#from model_ResGRU import NetPred
#from model_ResLSTM import NetPred
from model_MSCG2 import NetPred

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Load data    
    dataDir = os.path.abspath(os.path.dirname(__file__))
    #dataDir = r'D:\论文撰写\草稿——DSTnet for predicting VIVs\codes'
    #mat = scipy.io.loadmat(dataDir + '/data/Sample_VBI_ANSYS_Solid_Fixr1_001.mat')
    mat = scipy.io.loadmat(dataDir + '/data/Sample_Experi m=10.mat')

    # the training set contains training and validating samples
    x = mat['Input']  # vehicle wheel - effective excitation
    y = mat['Target']  # dynamic responses

    # basic information of samples
    num_sa = y.shape[0]
    num_inp = x.shape[2]  # the number of input features
    num_out = y.shape[2]  # number of output features, 5

    # hyperparameters
    epochs = 5000
    batch_size_train = 100
    batch_size_validate = 50

    # Scale data
    x_flat = np.reshape(x, [x.shape[0] * x.shape[1] * x.shape[2], 1])
    y_flat = np.reshape(y, [y.shape[0] * y.shape[1], y.shape[2]])

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    scaler_x.fit(x_flat)
    scaler_y.fit(y_flat)

    x_flat_scale = scaler_x.transform(x_flat)
    y_flat_scale = scaler_y.transform(y_flat)

    x_scale = np.reshape(x_flat_scale, [x.shape[0], x.shape[1], x.shape[2]])
    y_scale = np.reshape(y_flat_scale, [y.shape[0], y.shape[1], y.shape[2]])

    joblib.dump(scaler_x, dataDir +
                r'/results/scaler_x.save')
    joblib.dump(scaler_y, dataDir +
                r'/results/scaler_y.save')

    # generate the neural network
    net = NetPred(num_inp,num_out,200)
    net.to(device)
    
    loss_best = 1
    loss_train = []
    loss_validate = []
    
    loss_func = nn.MSELoss()
    
    optimizer_net = Adam(net.parameters(), lr=0.001, weight_decay=0)
    scheduler = lr_scheduler.StepLR(optimizer_net, step_size=5000, gamma=0.99)
    
    train_validate_indices = list(range(x.shape[0]))
    shuffle(train_validate_indices)
    fold_num = 10
    k_fold_num_sa = int(np.floor(x.shape[0]//fold_num))
    k_fold = 1
    k_fold_step = epochs//fold_num
    #train_indices = train_validate_indices[0:round(ratio_split * x.shape[0])]
    #validate_indices = train_validate_indices[round(ratio_split * x.shape[0]):]
    
    start = time.time()

    for e in range(epochs):
        # print('epoch = ', e + 1)
        # train
        net.train()
        
        if (e+1)%k_fold_step==1:
            validate_indices = train_validate_indices[(k_fold-1) * k_fold_num_sa:k_fold * k_fold_num_sa]
            train_indices = list(set(train_validate_indices)-set(validate_indices))
            k_fold += 1

        x_train = x_scale[train_indices]
        y_train = y_scale[train_indices]  # (35, 500, 5)
        x_validate = x_scale[validate_indices]
        y_validate = y_scale[validate_indices]  # (15, 500, 5)
        x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
        x_validate = torch.as_tensor(x_validate, dtype=torch.float32, device=device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
        y_validate = torch.as_tensor(y_validate, dtype=torch.float32, device=device)

        # train
        train_data = Data.TensorDataset(x_train, y_train)
        trainloader = Data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
        test_data = Data.TensorDataset(x_validate, y_validate)
        testloader = Data.DataLoader(test_data, batch_size=batch_size_validate, shuffle=True)
        net.train()
        loss = 0

        for ii, (x_train_batch, y_train_batch) in enumerate(trainloader):
            
            batch_size_train = x_train_batch.shape[0]

            optimizer_net.zero_grad()
            y_train_batch_pred = net(x_train_batch)
            loss_train_batch = loss_func(y_train_batch_pred, y_train_batch)
            loss_train_batch.backward()
            optimizer_net.step()
            scheduler.step()
            loss += loss_train_batch
        loss_train.append(loss.item() / (ii+1))

        # validate
        net.eval()
        loss = 0
        with torch.no_grad():
            for ii, (x_validate_batch, y_validate_batch) in enumerate(testloader):
                
                batch_size_validate = x_validate_batch.shape[0]
                
                y_validate_batch_pred = net(x_validate_batch)
                loss_validate_batch = loss_func(y_validate_batch_pred, y_validate_batch)
                loss += loss_validate_batch
            loss_validate.append(loss.item() / (ii+1))

        print('[epoch %d] loss_train: %.7f  loss_validate: %.7f' %
              (e + 1, loss_train[-1], loss_validate[-1]))

        if loss_validate[-1] < loss_best:
            loss_best = loss_validate[-1]
            torch.save(net.state_dict(), dataDir +
                       r'/results/best_model.h5')

    end = time.time()
    running_time = (end - start) / 3600
    print('Running Time: ', running_time, ' hour')
    print('Finished Training')

    # Plot training and testing loss
    plt.figure()
    plt.yscale('log')
    plt.plot(np.array(loss_train), 'b-', label='Training')
    plt.plot(np.array(loss_validate), 'm-', label='Validating')
    plt.legend()
    plt.show()

    scipy.io.savemat(dataDir + r'/results/results.mat',
                     {'loss_train': loss_train, 'loss_validate': loss_validate, 'loss_best': loss_best,
                      'running_time': running_time, 'epochs': epochs})


if __name__ == '__main__':
    main()
