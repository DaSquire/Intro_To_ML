#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:56:55 2023

@author: piripuz
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
from toolbox_02450 import rlr_validate
from sklearn import model_selection

replace = {
    'B':-1,
    'M':1}

df = pd.read_csv("../../Prostate_Cancer.csv", index_col="id")
ydf = df.diagnosis_result
X = df.replace(replace).to_numpy()

y = X[:, 0]
X = X[:, 1:]
X = (X - X.mean(axis=0))/X.std(axis=0)
X = np.insert(X, 0, 1, axis=1)

k=5
k2=10
CV = model_selection.KFold(k)

error = np.empty((k,1))
true_positivee = np.empty((k,1))
true_negativee = np.empty((k,1))
false_positivee = np.empty((k,1))
false_negativee = np.empty((k,1))
ls = np.empty((k,1))
err = np.empty((k,1))
lambdas = np.logspace(-1, 2, 15)
error_inner = np.empty((k2, len(lambdas)))
error_train_inner = np.empty((k2, len(lambdas)))

for i, (train_index, test_index) in enumerate(CV.split(X,y)):
    X_test = X[test_index]
    X_train = X[train_index]
    y_test = y[test_index]
    y_train = y[train_index]
    
    err[i], ls[i], _, error_inner[i], error_train_inner[i] = rlr_validate(X_train, y_train, lambdas)
    l = ls[i]
    w = np.linalg.solve((X_train.T @ X_train + l*np.identity(X.shape[1])), X_train.T @ y_train)
    y_test_predict_proba = 1/(1+np.exp(-(X_test @ w)))
    y_test_predict = (np.rint(y_test_predict_proba)-0.5)*2
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

fig = plt.figure(figsize=(6,4),dpi=300)
ax = fig.subplots(1,1)
# plt.yticks(np.arange(0, 0.5, 0.05))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
ax.set_ylabel('Error')
ax.set_xlabel('Regularization parameter')
fold = 2
plt.plot(lambdas, error_train_inner[fold,:], 'bo-')
plt.plot(lambdas, error_inner[fold,:], 'ro-')
ax.set_xscale('log')
plt.legend([ 'Error on test data', 'Error on train data'])
dire = '/home/piripuz/Università/Magistrale/mach_learning/Progetto/2/6537d0e3efd6b3805617a6ab/images/{0}'
try:
    plt.savefig(dire.format('logistic_par'))
except:
    print("Directory {0} not found".format(dire), bbox_inches='tight')

#%%

y_predict_proba = 1/(1+np.exp(-(X @ w)))

palette= {'M':[1, 0, 0], 'B':[0, 1, 0]}
marker = {'train': '+', 'test': 'o'}
color = np.array([palette[x] for x in ydf])
y_j = (-y+1)/2 + np.random.default_rng(123).normal(scale=0.12, size=len(y))
fig, ax = plt.subplots(figsize=(10,8), dpi=400)
plt.text(1, 0.75, format(false_positive, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[1,0,0,0.5])
plt.text(1, 0.25, format(true_negative, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[0,1,0,0.5])
plt.text(0, 0.75, format(true_positive, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[0,1,0,0.5])
plt.text(0, 0.25, format(false_negative, '.0%'), fontsize=50, horizontalalignment='center', verticalalignment='center', color=[1,0,0,0.5])
plt.scatter(y_j[train_index], y_predict_proba[train_index], c=color[train_index], marker=marker['train'])
plt.scatter(y_j[test_index], y_predict_proba[test_index], c=color[test_index], marker=marker['test'], edgecolors='black')
red_ball = mlines.Line2D([], [], color=[1,0,0], marker='o', linestyle='None', markeredgecolor='black',
                           label='Malignant')
green_ball = mlines.Line2D([], [], color=[0,1,0], marker='o', linestyle='None', markeredgecolor='black',
                           label='Benign')
circle = mlines.Line2D([], [], color=[0, 0, 0], marker=marker['test'], linestyle='None',
                           label='Test set')
cross = mlines.Line2D([], [], color=[0, 0, 0], marker=marker['train'], linestyle='None',
                           label='Train set')
leg = ax.legend(title='Diagnosis result', handles=[red_ball, green_ball],
                          loc='upper right')
ax.add_artist(leg)
ax.legend(title='Set', handles=[circle, cross], loc='lower right')
ax.spines['right'].set_position(('data', 0.5))
ax.spines['bottom'].set_position(('data', 0.5))
ax.spines['top'].set_color('none')
# # plt.xlim([-1.2, 1.2])
plt.ylim([0, 1])
plt.xticks([])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
c = (np.array([[false_negative, true_negative],
                  [true_positive, false_positive]])*255).\
                        astype('int64')

z = np.zeros((2,2,3), dtype='int64')
z[:,:,0] = [[128, 51], [51, 128]]
z[:,:,1] = [[0, 102], [102, 0]]
z[:,:,2] = [[0, 0], [0, 0]]
c = np.insert(z, 3, c, axis=2)
plt.imshow(c)
dire = '/home/piripuz/Università/Magistrale/mach_learning/Progetto/2/6537d0e3efd6b3805617a6ab/images/{0}'
try:
    plt.savefig(dire.format('confusion_logistic'), bbox_inches='tight')
except:
    print("Directory {0} not found".format(dire))