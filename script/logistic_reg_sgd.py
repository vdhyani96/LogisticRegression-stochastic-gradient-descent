# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def stoch_grd_dcnt(train, batch_size, n0, n1, max_epoch, delta):
    # getting the total number of classes
    k = train['Category'].nunique()
    theta = np.zeros((train.shape[1], k-1))
    theta_list = []
    theta_list.append(theta)
    loss_list = []
    loss_list.append(loss(theta, train))
    for epoch in range(max_epoch):
        print(epoch+1, "Epochs started.")
        n = n0/(n1 + epoch)
        train = train.sample(frac=1, random_state = 102)
        # dividing the training set into batches
        num_batch = int(train.shape[0]/batch_size)
        batches = np.split(train, num_batch, axis = 0)
        # theta update using gradient descent
        for i in range(num_batch):
            theta = np.array(grad_desc(np.array(theta), n, batches[i], k))
        theta_list.append(theta)
        # computing and storing the loss function using new theta
        loss_list.append(loss(theta, train))
        if(loss_list[-1] > (1 - delta) * loss_list[-2] and 
           epoch > 2):
            print("Not much progress, terminate")
            break
        print(epoch+1, "Epochs completed.")
    return theta_list, loss_list


def grad_desc(theta, n, batch, k):
    y = batch['Category']
    batch = batch.drop(['Category'], axis = 1)
    batch_size = batch.shape[0]
    # adding the bias term
    batch['bias'] = [1 for i in range(batch_size)]
    indf = -1
    for i in range(k-1):
        derivative = 0
        for j in range(batch_size):
            if(y.iloc[j] == (i+1)):
                indf = 1
            else:
                indf = 0
            xi = batch.iloc[j, :].T
            prob = calcprob(i, theta, xi, False)
            diff = indf - prob
            dv = diff * xi
            derivative += dv
        final_dv = (-1) * derivative / batch_size
        theta[:, i] = theta[:, i] - (n * final_dv)
    return theta


# calculate the probability of xi to be of class cval+1
# flag = false for 1-(k-1) and flag = true for kth class
def calcprob(cval, theta, xi, flag):
    denom = 0
    for i in range(theta.shape[1]):
        denom += np.exp(np.matmul(theta[:, i], xi))
    denom += 1
    denom = denom
    if flag is False:
        num = np.exp(np.matmul(theta[:, cval].T, xi))
        num = num
    else:
        num = 1
    res = num/denom
    return res


# loss function for logistic regression
def loss(theta, x):
    y = x['Category']
    x = x.drop(['Category'], axis = 1)
    x['bias'] = [1 for i in range(x.shape[0])]
    logloss = 0
    num = x.shape[0]
    k = y.nunique()
    for i in range(num):
        xi = x.iloc[i, :].T
        proby = []
        for j in range(k-1):
            prob = calcprob(j, theta, xi, False)
            proby.append(prob)
        # now for kth class
        prob = calcprob(k, theta, xi, True)
        proby.append(prob)
        prob = np.max(proby)
        logprob = np.log(prob)
        logloss += logprob
    finalloss = (-1) * (logloss/num)
    return finalloss
        
            
# Function to standardize data
def standardize(df):
    for i in range(df.shape[1]):
        vector = df.iloc[:, i]
        mean = np.mean(vector)
        std = np.std(vector)
        df.iloc[:, i] = (df.iloc[:, i] - mean)/std
    return df



# assumes test is in normal tabular form
def predict(theta, test):
    test = test.T
    pred_matrix = np.zeros((4, test.shape[1]))
    for i in range(test.shape[1]):
        xi = test.iloc[:, i]
        for j in range(3):
            prob = calcprob(j, theta, xi, False)
            pred_matrix[j, i] = prob
        pred_matrix[3, i] = calcprob(3, theta, xi, True)
    return pred_matrix
    


# reading the feature dataset
Train_Features = pd.read_pickle('Train_Features.pkl', compression="infer")
Train_Feat = pd.DataFrame.from_dict(Train_Features, orient='index')
# standardizing the training features
Train_Feat = standardize(Train_Feat)
Train_Feat['Id'] = Train_Feat.index

Val_Features = pd.read_pickle('Val_Features.pkl', compression="infer")
Val_Feat = pd.DataFrame.from_dict(Val_Features, orient='index')
# standardize the validation features
Val_Feat = standardize(Val_Feat)
Val_Feat['Id'] = Val_Feat.index

Test_Features = pd.read_pickle('Test_Features.pkl', compression="infer")
Test_Feat = pd.DataFrame.from_dict(Test_Features, orient='index')

# reading the labels
Train_labels = pd.read_csv('Train_Labels.csv')
Val_labels = pd.read_csv('Val_Labels.csv')

# joining labels with the feature set
Train = Train_Feat.merge(Train_labels, how = 'inner', on = 'Id')
Train = Train.drop(['Id'], axis = 1)

Val = Val_Feat.merge(Val_labels, how = 'inner', on = 'Id')
Val = Val.drop(['Id'], axis = 1)

# 1. Run the first stochastic gradient descent on the training example
params = stoch_grd_dcnt(Train, 16, 0.1, 1, 1000, 0.00001)
theta_list = params[0]
loss_list = params[1]
#>> Number of epochs = 5
#>> Final value of L(theta) = 0.49066431813575173
# Plotting the L(theta) against the number of epochs
plt.plot(loss_list)
plt.grid()

# 2. Experimenting with a few values of (n0, n1) for sgd learning rate
params1 = stoch_grd_dcnt(Train, 16, 0.01, 1, 1000, 0.00001)
params2 = stoch_grd_dcnt(Train, 16, 0.05, 1, 1000, 0.00001)
params3 = stoch_grd_dcnt(Train, 16, 0.5, 1, 1000, 0.00001)
params4 = stoch_grd_dcnt(Train, 16, 1, 1, 1000, 0.00001)
#>> Values of n0, n1 with fastest convergence = 1, 1 (Params4)
#>> Number of epochs = 4
#>> Final value of L(theta) = 0.09567952584634766
# Plotting the L(theta) with the epochs
plt.plot(params4[1])
plt.grid()

# 3. Run the algorithm on the validation data
params_val = stoch_grd_dcnt(Val, 16, 0.1, 1, 1000, 0.00001)
# plotting
plt.plot(params_val[1])
plt.plot(params[1])
plt.legend(['Loss with Validation', 'Loss with Training'], loc = 'upper right')
plt.grid()
# now calculating the accuracies
train_accuracy, val_accuracy = [], []
# for training data
tr_label = Train['Category']
Train = Train.drop(['Category'], axis = 1)
# append bias
Train['bias'] = [1 for i in range(Train.shape[0])]

for i in range(len(params[0])):
    theta = params[0][i]
    pred = predict(theta, Train)
    predictions = np.argmax(pred, axis = 0)
    predictions = predictions + 1
    matches = np.sum(predictions == tr_label)
    accuracy = matches/4000
    train_accuracy.append(accuracy)

# for validation data
val_label = Val['Category']
Val = Val.drop(['Category'], axis = 1)
# append bias
Val['bias'] = [1 for i in range(Val.shape[0])]

for i in range(len(params_val[0])):
    theta = params_val[0][i]
    pred = predict(theta, Val)
    predictions = np.argmax(pred, axis = 0)
    predictions = predictions + 1
    matches = np.sum(predictions == val_label)
    accuracy = matches/2000
    val_accuracy.append(accuracy)

# now plotting the accuracies
plt.plot(val_accuracy)
plt.plot(train_accuracy)
plt.legend(['Validation accuracy', 'Training accuracy'])
plt.grid()


# 4. Confusion matrices for validation and training sets
# first for training set
theta = params[0][5]
pred = predict(theta, Train)
predictions = np.argmax(pred, axis = 0)
predictions = predictions + 1
tr_confusion = pd.crosstab(tr_label, predictions)
print(tr_confusion)
#Predicted   1    2    3    4
#Actual                    
#1.0       457  253   40   33
#2.0       136  879  140  114
#3.0        62  262  319  173
#4.0        40  133  137  822

# now for validation set
theta = params_val[0][6]
pred = predict(theta, Val)
predictions = np.argmax(pred, axis = 0)
predictions = predictions + 1
val_confusion = pd.crosstab(val_label, predictions)
print(val_confusion)
#Predicted   1    2    3    4
#Actual                    
#1.0       213   72   13   18
#2.0        60  487   62   51
#3.0        10  119  225   84
#4.0         7   56   50  473




### For submitting to kaggle
Train_Feat = pd.DataFrame.from_dict(Train_Features, orient='index')
Val_Feat = pd.DataFrame.from_dict(Val_Features, orient='index')
Test_Feat = pd.DataFrame.from_dict(Test_Features, orient='index')
# merge these datasets together before standardizing
dat = pd.concat([Train_Feat, Val_Feat, Test_Feat], ignore_index = False)
# standardizing dat
dat = standardize(dat)
test = dat.iloc[6000:,:]
train = dat.iloc[:6000, :]
train['Id'] = train.index
# join labels with the feature set
Train_labels = pd.read_csv('Train_Labels.csv')
Val_labels = pd.read_csv('Val_Labels.csv')
labels = pd.concat([Train_labels, Val_labels], ignore_index = False)
train = train.merge(labels, how = 'inner', on = 'Id')
train = train.drop(['Id'], axis = 1)

# run the algorithm with sgd
test_params = stoch_grd_dcnt(train, 16, 0.1, 1, 10, 0.00001)
# append bias at the end of the test set
test['bias'] = [1 for i in range(test.shape[0])]
theta = test_params[0][1]
pred = predict(theta, test)
# predicting the class from the obtained prediction matrix
predictions = np.argmax(pred, axis = 0)
predictions = predictions + 1
predictions = np.reshape(predictions, (2000, 1))

# creating my first submission
#id_no = np.array([str(i+1) for i in range(predictions.shape[0])]).T
#id_no = np.reshape(id_no, (2000, 1))
id_num = np.array(test.index.tolist())
id_num = np.reshape(id_num, (2000, 1))
submission = np.append(id_num, predictions, axis = 1)
df = pd.DataFrame(data = submission, columns = ['Id', 'Category'])
df.to_csv('submission.csv', index = False)

# Accuracy on Kaggle: 0.37833
