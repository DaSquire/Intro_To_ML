#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:41:15 2023

@author: piripuz
"""

import numpy as np
from sklearn import tree
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt

replace = {
    'B':-1,
    'M':1}

df = pd.read_csv("../../Prostate_Cancer.csv", index_col="id")
ydf = df.diagnosis_result
X = df.replace(replace).to_numpy()
df = df.drop('diagnosis_result', axis=1)
y = X[:, 0]
X = X[:, 1:]
X = (X - X.mean(axis=0))/X.std(axis=0)

pars = np.rint(np.linspace(4, 30, 12)).astype('int64')
k1 = 10
k2 = 10
CV_outer = model_selection.KFold(k1, shuffle=True, random_state=33)
CV_inner = model_selection.KFold(k2, shuffle=True, random_state=396555)

error = np.empty((k1,1))
true_positivee = np.empty((k1,1))
true_negativee = np.empty((k1,1))
false_positivee = np.empty((k1,1))
false_negativee = np.empty((k1,1))
best_pars = [0]*k1

for i, (train_index, test_index) in enumerate(CV_outer.split(X, y)):
    X_test = X[test_index]
    X_train = X[train_index]
    y_test = y[test_index]
    y_train = y[train_index]
    error_inner = np.empty((k2, len(pars)))
    for j, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train, y_train)):
        X_test_inner = X[test_index_inner]
        X_train_inner = X[train_index_inner]
        y_test_inner = y[test_index_inner]
        y_train_inner = y[train_index_inner]
        for p in range(len(pars)):
            tr = tree.DecisionTreeClassifier(min_samples_split=pars[p], random_state=452)
            tr.fit(X_train_inner, y_train_inner)
            y_test_predict_inner = tr.predict(X_test_inner)
            error_inner[j][p] = sum(y_test_predict_inner!=y_test_inner)/len(y_test_inner)
    mean_err = np.mean(error_inner, axis=0)
    best_pars[i] = pars[np.argmin(mean_err)]
    best_par = best_pars[i]
    best_tree = tree.DecisionTreeClassifier(min_samples_leaf=best_par, random_state=12)
    best_tree.fit(X, y)
    y_test_predict = best_tree.predict(X_test)
    
    error[i] = sum(y_test_predict != y_test)/len(y_test)
    
    true_positivee[i] = np.count_nonzero((y_test_predict[y_test==1])==1)/(y_test[y_test==1]).size
    true_negativee[i] = np.count_nonzero((y_test_predict[y_test==-1])==-1)/(y_test[y_test==-1]).size
    false_positivee[i] = np.count_nonzero((y_test_predict[y_test==-1])==1)/(y_test[y_test==-1]).size
    false_negativee[i] = np.count_nonzero((y_test_predict[y_test==1])==-1)/(y_test[y_test==1]).size

print(np.mean(error))
true_positive = np.mean(true_positivee)
true_negative = np.mean(true_negativee)
false_positive = np.mean(false_positivee)
false_negative = np.mean(false_negativee)

#%%

fig, ax = plt.subplots(figsize=(8,8), dpi=400)
plt.text(1, 0.75, format(false_positive, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[1,0,0,0.5])
plt.text(1, 0.25, format(true_negative, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[0,1,0,0.5])
plt.text(0, 0.75, format(true_positive, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[0,1,0,0.5])
plt.text(0, 0.25, format(false_negative, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[1,0,0,0.5])
plt.ylim([0, 1])


ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')

plt.xticks([0, 1], ['Malignant', 'Benign'])
plt.yticks([0.25, 0.75], [ 'Reported\n benign', 'Reported\n malignant'])

# c = (np.array([[false_negative, true_negative],
#                   [true_positive, false_positive]])*255).\
#                         astype('int64')

# z = np.zeros((2,2,3), dtype='int64')
# z[:,:,0] = [[128, 51], [51, 128]]
# z[:,:,1] = [[0, 102], [102, 0]]
# z[:,:,2] = [[0, 0], [0, 0]]
# c = np.insert(z, 3, c, axis=2)
# plt.imshow(c)
fig = plt.figure(figsize=(4,4),dpi=300)
# tree.plot_tree(best_tree, feature_names=list(df.columns),\
#                class_names=['Benign', 'Malignant'], max_depth=1)
# plt.savefig('Tree.png', bbox_inches='tight')
#%%
plt.plot(pars, mean_err, 'ro-')
plt.legend(['Error in '])