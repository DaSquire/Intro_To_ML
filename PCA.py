#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 19:04:13 2023

@author: piripuz
"""

import pandas as pd
import numpy as np
import scipy.linalg as linalg
from matplotlib.pyplot import (figure, axes, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm, scatter, xticks, savefig)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
import array_to_latex as a2l
import seaborn as sns
import math
#%%
dict_r = {
        'B' : 'Benign',
        'M' : 'Malign'
        }
df = pd.read_csv("./Prostate_Cancer.csv", index_col='id')
df = df.replace(dict_r)
X=df.drop('diagnosis_result', axis=1).to_numpy()
N,M = X.shape

Xc = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
U,S,Vh = linalg.svd(Xc,full_matrices=False)  
V = Vh.T
Z = Xc @ V
rho = (S*S) / (S*S).sum()
#%%
fig = figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(rho,'o-')
plot(np.cumsum(rho),'x-', color='g')
xticks(np.arange(len(rho)), np.arange(1, len(rho)+1))
xlabel('Principal component')
ylabel('Variance explained')

palette = {
    'Benign': 'tab:green',
    'Malign': 'tab:red',
}
#%%
X_plot = np.linspace(50, 160, num=500)
Y_plot = np.vectorize((lambda x: x**2/(4*math.pi)))(X_plot)

area = X[:,3]
per = X[:,2]


sns.relplot(x=Z[:,0], y=Z[:,1], hue=df['diagnosis_result'], palette=palette)
err = sns.relplot(x=per,y=area, hue=df['diagnosis_result'], palette=palette)
plot(X_plot, Y_plot, color='b', alpha=0.5)
xlabel('Perimeter')
ylabel('Area')
err._legend.set_title('Diagnosis result')
savefig('../latex/images/size.pdf')
#%%
const = 0.06948074

X_plot = np.linspace(2500, 30000, num=500)
Y_plot = np.vectorize((lambda x: x*const))(X_plot)

corr2 = sns.relplot(x=per**2,y=area, hue=df['diagnosis_result'], palette=palette)
plot(X_plot, Y_plot, color='b', alpha=0.5)
xlabel('Perimeter squared')
ylabel('Area')
corr2._legend.set_title('Diagnosis result')
savefig('../latex/images/size2.pdf')

#%%
a_std = area - (per**2)*const
err = sns.relplot(x=per,y=a_std, hue=df['diagnosis_result'], palette=palette)
plot([50, 160], [0,0], color='b', alpha=0.5)
err._legend.set_title('Diagnosis result')
xlabel('Perimeter')
ylabel('Error')
savefig('../latex/images/sizeError.pdf')

#%%
Xnc = np.delete(Xc, 2, axis=1)
U,S,Vn = linalg.svd(Xnc,full_matrices=False)  
Vn = Vn.T
Z = Xnc @ Vn
rho2 = (S*S) / (S*S).sum()

#%%
fig = figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.cumsum(rho2),'o-')
xticks(np.arange(len(rho)), np.arange(1, len(rho)+1))
xlabel('Principal component')
ylabel('Variance explained')
savefig('../latex/images/pca2var.pdf')
savefig('../650de7553e6e3f99dbc97883/images/pca2var.pdf')


#%%
c = Vn[6,1]
sns.relplot(x=Z[:,1], y=Xnc[:,6])
plot([-3, 3], [-3*c, 3*c], color='b', alpha=0.5)
#%%

MM = np.empty([8,8])

for i in range(8):
    for j in range(8):
        MM[i,j] = (V[i,:]/V[j,:]).var()
# sns.pairplot(df, hue='diagnosis_result')
# title('Size over perimeter')
#%%
totVar = np.append(0, np.cumsum(rho))
fig, ax = (figure(figsize=[3, 4.8]), axes())
ax.get_xaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
for i in range(8):
    
    if(i<5):
        b = ax.bar(0, rho[i], width=0.2, bottom=totVar[i], label='PC'+str(i+1), color='C3', alpha=np.exp(-i/7))
        ax.bar_label(b, fmt='{:.0%}', label_type='center')
    else:
        b = ax.bar(0, rho[i], width=0.2, bottom=totVar[i], color='C3', alpha=np.exp(-i/7))
ax.legend()
ax.set_xlim(left=0, right=0.22)


'''three_d = plot()
plot_axes = three_d.axes(projection='3d')
plot_axes.scatter3D(Z[:,0],Z[:,1],Z[:,2])'''
#fig=figure()
#ax = Axes3D(fig, auto_add_to_figure=False)
#fig.add_axes(ax)
# sc = ax.scatter(Z[:,0], Z[:, 1], Z[:, 2], c=df['diagnosis_result'])