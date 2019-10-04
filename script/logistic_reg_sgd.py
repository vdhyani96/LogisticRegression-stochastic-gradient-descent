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
    for epoch in range(max_epoch):
        n = n0/(n1 + epoch)
        train = train.sample(frac=1, random_state = 102)
        # dividing the training set into batches
        batches = np.split(train, batch_size, axis = 0)
        







os.chdir('C:\\Users\\admin\\Desktop\\PostG\\GRE\\Second Take\\Applications\\Univs\\Stony Brook\\Fall 19 Courses\\ML\\Homeworks\\Homework 3\\Kaggle')
# reading the feature dataset
Train_Features = pd.read_pickle('Train_Features.pkl', compression="infer")
Train_Feat = pd.DataFrame.from_dict(Train_Features, orient='index')

Val_Features = pd.read_pickle('Val_Features.pkl', compression="infer")
Val_Feat = pd.DataFrame.from_dict(Val_Features, orient='index')

Test_Features = pd.read_pickle('Test_Features.pkl', compression="infer")
Test_Feat = pd.DataFrame.from_dict(Test_Features, orient='index')

# reading the labels
Train_labels = pd.read_csv('Train_Labels.csv')
Val_labels = pd.read_csv('Val_Labels.csv')

# joining labels with the feature set -- will do later, join on index






