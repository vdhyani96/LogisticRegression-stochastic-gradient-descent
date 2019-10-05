# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
import pickle



def stoch_grd_dcnt(train, batch_size, n0, n1, max_epoch, delta):
    # getting the total number of classes
    k = train['Category'].nunique()
    theta_old = np.zeros((train.shape[1], k))
    theta_new = theta_old
    for epoch in range(max_epoch):
        print(epoch, "Epochs started.")
        n = n0/(n1 + epoch)
        train = train.sample(frac=1, random_state = 102)
        # dividing the training set into batches
        num_batch = train.shape[0]/batch_size
        batches = np.split(train, num_batch, axis = 0)
        
        # theta update using gradient descent
        for i in range(num_batch):
            theta_new = grad_desc(theta_new, n, batches[i])
        
        if(theta_new>theta_old):
            break
        else:
            theta_old = theta_new
        
        print(epoch, "Epochs completed.")


def grad_desc(theta, n, batch):
    




os.chdir('C:\\Users\\admin\\Desktop\\PostG\\GRE\\Second Take\\Applications\\Univs\\Stony Brook\\Fall 19 Courses\\ML\\Homeworks\\Homework 3\\Kaggle')
# reading the feature dataset
Train_Features = pd.read_pickle('Train_Features.pkl', compression="infer")
Train_Feat = pd.DataFrame.from_dict(Train_Features, orient='index')
Train_Feat['Id'] = Train_Feat.index

Val_Features = pd.read_pickle('Val_Features.pkl', compression="infer")
Val_Feat = pd.DataFrame.from_dict(Val_Features, orient='index')
Val_Feat['Id'] = Val_Feat.index

Test_Features = pd.read_pickle('Test_Features.pkl', compression="infer")
Test_Feat = pd.DataFrame.from_dict(Test_Features, orient='index')

# reading the labels
Train_labels = pd.read_csv('Train_Labels.csv')
Val_labels = pd.read_csv('Val_Labels.csv')

# joining labels with the feature set
Train = Train_Feat.merge(Train_labels, how = 'inner', on = 'Id')
Train = Train.drop(['Id'], axis = 1)




