#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:27:18 2023

@author: piripuz
"""

import numpy as np
from sklearn import tree
import pandas as pd
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from scipy.stats import beta, binom
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick

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

pars = np.rint(np.linspace(2, 12,10)).astype('int64')
lambdas = np.logspace(-1, 2, 15)
k1 = len(X)
k2 = 10
CV_outer = model_selection.KFold(k1, shuffle=True, random_state=33)
CV_inner = model_selection.KFold(k2, shuffle=True, random_state=396555)
n11_arr = np.empty(k1)
n10_arr = np.empty(k1)
n01_arr = np.empty(k1)
n00_arr = np.empty(k1)
err_baseline = np.empty(k1)
err_log = np.empty(k1)
err_tree = np.empty(k1)
best_lambdas = np.empty(k1)
best_pars = np.empty(k1, dtype='int64')

for i, (train_index, test_index) in enumerate(CV_outer.split(X, y)):
    X_test = X[test_index]
    X_train = X[train_index]
    y_test = y[test_index]
    y_train = y[train_index]
    error_inner = np.empty((k2, len(pars)))
    error_train_inner = np.empty((k2, len(pars)))
    baseline = -1 if sum(y_train == -1) >= len(y_train)/2 else 1
    for j, (train_index_inner, test_index_inner) in enumerate(CV_inner.split(X_train, y_train)):
        X_test_inner = X[test_index_inner]
        X_train_inner = X[train_index_inner]
        y_test_inner = y[test_index_inner]
        y_train_inner = y[train_index_inner]
        for p in range(len(pars)):
            tr = tree.DecisionTreeClassifier(min_samples_leaf=pars[p], random_state=4512)
            tr.fit(X_train_inner, y_train_inner)
            y_test_predict_inner = tr.predict(X_test_inner)
            y_train_predict_inner = tr.predict(X_train_inner)
            error_inner[j][p] = sum(y_test_predict_inner!=y_test_inner)/len(y_test_inner)
            error_train_inner[j][p] = sum(y_train_predict_inner!=y_train_inner)/len(y_train_inner)
        mean_err = np.mean(error_inner, axis=0)
        best_pars[i] = pars[np.argmin(mean_err)]
    best_tree = tree.DecisionTreeClassifier(min_samples_leaf=best_pars[i], random_state=12)
    best_tree.fit(X, y)
    err, best_lambdas[i], _, _, _ = rlr_validate(X_train, y_train, lambdas, cvf=k2)
    w = np.linalg.solve((X_train.T @ X_train + best_lambdas[i]*np.identity(X.shape[1])), X_train.T @ y_train)
    y_test_predict_proba = 1/(1+np.exp(-(X_test @ w)))
    y_test_predict_log = (np.rint(y_test_predict_proba)-0.5)*2
    y_test_predict_tree = best_tree.predict(X_test)
    y_test_predict_baseline = [baseline] * len(y_test)
    
    err_tree[i] = (y_test_predict_tree!=y_test).sum()/len(y_test)
    err_log[i] = (y_test_predict_log!=y_test).sum()/len(y_test)
    err_baseline[i] = (y_test_predict_baseline!=y_test).sum()/len(y_test)
        
    n11_arr[i] = sum((y_test_predict_baseline == y_test) & (y_test_predict_tree == y_test))
    n00_arr[i] = sum((y_test_predict_baseline != y_test) & (y_test_predict_tree != y_test))
    n01_arr[i] = sum((y_test_predict_baseline == y_test) & (y_test_predict_tree != y_test))
    n10_arr[i] = sum((y_test_predict_baseline != y_test) & (y_test_predict_tree == y_test))
n11 = np.sum(n11_arr)
n10 = np.sum(n10_arr)
n01 = np.sum(n01_arr)
n00 = np.sum(n00_arr)
table = pd.DataFrame(data=np.array([best_lambdas, err_log, best_pars, err_tree, err_baseline]).T,\
                     columns=['$\lambda_i$', '$E_i^{test}$', '$p_i^*$', '$E_i^{test}$', '$E_i^{test}$' ],\
                         index=np.arange(1, k1+1, 1)).round(2)
table.index.name = '$i$'
table = table.astype(str)

#%%
fig = plt.figure(figsize=(6,4),dpi=300)
ax = fig.subplots(1,1)
plt.yticks(np.arange(0, 0.5, 0.05))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
ax.set_ylabel('Error')
ax.set_xlabel('Max sample for splitting')
fold = 1
plt.plot(pars, np.mean(error_inner, axis=0), 'ro-')
plt.plot(pars, np.mean(error_train_inner, axis=0), 'bo-')
plt.legend([ 'Error on test data', 'Error on train data'])
plt.savefig('Tree_par')
dire = '/home/piripuz/Università/Magistrale/mach_learning/Progetto/2/6537d0e3efd6b3805617a6ab/images/{0}'
try:
    plt.savefig(dire.format('tree_par'))
except:
    print("Directory {0} not found".format(dire), bbox_inches='tight')
#%%
n = len(y_train)
theta_est = (n10-n01)/n
Q = n*n*(n+1)*(theta_est+1)*(1-theta_est)/(n*(n10+n01)-(n10-n01)**2)
f = (theta_est + 1)*(Q-1)/2
g = (-theta_est + 1)*(Q-1)/2
lx = 50
low = [0]*lx
high = [0]*lx
betas = np.logspace(-3, -0.4, lx)
for i, alpha in enumerate(betas):
    lower = 2*beta.ppf(alpha/2, f, g) - 1
    low[i] = lower
    higher = 2*beta.ppf(1-alpha/2, f, g) - 1
    high[i] = higher
p = 2*binom.cdf(min(n01, n10), n10+n01, 1/2)
#%%
fig = plt.figure(figsize=(6,4), dpi = 400)
ax = fig.add_subplot(1, 1, 1)
plt.axline((0,0),slope=0, color='violet')
plt.axline((0,theta_est),slope=0, color='blue')
plt.plot(betas, list(zip(list(high), list(low))), 'r')
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
ax.set_xlabel('Beta', loc='center')
ax.set_ylabel('Theta')
plt.xticks(np.arange(0, max(betas), 0.05), rotation=45)
plt.fill_between(betas, high, low, color='green')
green_patch = mpatches.Patch(color='green')
plt.legend(handles=[green_patch, ax.lines[2], ax.lines[1]],\
           labels=['Confidence interval', 'Confidence interval bounds', 'Expected accuracy difference'],\
               loc='best')
dire = '/home/piripuz/Università/Magistrale/mach_learning/Progetto/2/6537d0e3efd6b3805617a6ab/images/{0}'
try:
    plt.savefig(dire.format('confidence_interval'), bbox_inches='tight')
except:
    print("Directory {0} not found".format(dire))