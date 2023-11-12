#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:55:11 2023

@author: piripuz
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines

replace = {
    'B':-1,
    'M':1}

df = pd.read_csv("Prostate_Cancer.csv", index_col="id")
ydf = df.diagnosis_result
X = df.replace(replace).to_numpy()

y = X[:, 0]
X = X[:, 1:]
X = (X - X.mean(axis=0))/X.std(axis=0)
X = np.insert(X, 0, 1, axis=1)

rnd = np.random.default_rng(1234)
test_index = rnd.choice(range(len(X)), 20, replace=False)
train_index = np.array([i for i in range(0, len(X)) if i not in test_index])

X_test = X[test_index]
X_train = X[train_index]
y_test = y[test_index]
y_train = y[train_index]

#%%

w = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
y_predict_proba = 1/(1+np.exp(-(X @ w)))
y_predict = np.rint(y_predict_proba)
err = np.abs((y+1)/2-y_predict)

#%%
error_train = err[train_index].mean()
error_test = err[test_index].mean()

y_predict_test = y_predict[test_index]

true_positive = np.count_nonzero((y_predict_test[y_test==1])==1)/(y_test[y_test==1]).size
true_negative = np.count_nonzero((y_predict_test[y_test==-1])==0)/(y_test[y_test==-1]).size
false_positive = np.count_nonzero((y_predict_test[y_test==-1])==1)/(y_test[y_test==-1]).size
false_negative = np.count_nonzero((y_predict_test[y_test==1])==0)/(y_test[y_test==1]).size

#%%
palette= {'M':[1, 0, 0], 'B':[0, 1, 0]}
marker = {'train': '+', 'test': 'o'}
color = np.array([palette[x] for x in ydf])
y_j = (-y+1)/2 + np.random.default_rng(123).normal(scale=0.12, size=len(y))
fig, ax = plt.subplots(figsize=(8,8), dpi=400)
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
plt.savefig('./conf_mat.pdf', format='pdf', bbox_inches='tight')