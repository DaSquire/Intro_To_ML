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

df = pd.read_csv("prostate.csv")