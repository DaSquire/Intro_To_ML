import pandas as pd
import numpy as np
import scipy.linalg as linalg
from matplotlib.pyplot import (figure, subplots, axes, subplot, plot, xlabel, ylabel, title, 
yticks, show,legend,imshow, cm, scatter, xticks, savefig)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
import array_to_latex as a2l
import seaborn as sns
import math

df = pd.read_csv("Prostate_Cancer.csv", index_col="id")
df["diagnosis_result"] = np.where(df["diagnosis_result"] == "M", 1, 0)

continuous_cols = ['radius', 
                    'texture', 
                    'perimeter', 
                    'area', 
                    'smoothness', 
                    'compactness', 
                    'symmetry', 
                    'fractal_dimension']

df_means = df[continuous_cols].mean()

df_std = df[continuous_cols].std()

df[continuous_cols] = (df[continuous_cols] -  df_means) / df_std

X = df.to_numpy()

print(X)