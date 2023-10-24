#


import toolbox_02450
# print(toolbox_02450.__version__)
import xlrd
import numpy as np
import seaborn as sns

from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
doc = xlrd.open_workbook('../Data/Prostate_Cancer.xls').sheet_by_index(0)

#Preallocate memory, then extract excel data to matrix X
X = np.empty((100, 10))
for i, col_id in enumerate(range(0, 10)):
    column_data = np.asarray(doc.col_values(col_id, 1, 101))

    # If this is the second column (index 1), replace 'M' with 1 and 'B' with 0
    if i == 1:
        column_data = np.where(column_data == 'M', 1, 0)
    if i==4 :
        print(column_data)
        column_data = column_data**2
        print(column_data)
    X[:, i] = column_data

# print(X)
missing_values = np.isnan(X)
# print(missing_values)
if np.any(missing_values):
    print("The data contains missing values.")
else:
    print("The data does not contain missing values.")


# exercise 3.2.1
# Compute values mean,standard deviation,median,range
# for index in range(1, 10):

#     print(doc.row_values(0, index, index+1))
#     mean_x = X[:,index].mean()
#     std_x = X[:,index].std(ddof=1)
#     median_x = np.median(X[:,index])
#     range_x = X[:,index].max()-X[:,index].min()

#     # Display results
#     # print('Vector:',X[:,i])
#     print('Mean:',mean_x)
#     print('Standard Deviation:',std_x)
#     print('Median:',median_x)
#     print('Range:',range_x)
#     print(" Space between values")
# For all features excluding diagnostic result
# plt.figure(figsize = (20, 10))
# plotnumber = 1
# for column in range(10):
#     if column!=1 and column!=0:
#         if plotnumber <= 14:
#             ax = plt.subplot(3, 5, plotnumber)
#             sns.distplot(X[:,column],color='blue',)
#             plt.xlabel(doc.row_values(0,column,column+1))
        
        
#         plotnumber += 1

# plt.tight_layout()
# plt.show() 

# diagnostic result
# plt.figure(figsize=(20, 15))
# plotnumber = 1

# if plotnumber <= 11:
#     ax = plt.subplot(3, 4, plotnumber)
#     sns.countplot(x=X[:, 1], palette='rocket', color='black')
#     plt.xlabel(doc.row_values(0, 1, 2)[0])  # Get the column name from the Excel sheet

# plotnumber += 1

# plt.tight_layout()
# plt.show()
columns_to_exclude = [0,1]  # List of column indices to exclude

# a new matrix X excluding the specified columns
X_filtered = np.delete(X, columns_to_exclude, axis=1)

# the covariance matrix
cov_matrix = np.cov(X_filtered, rowvar=False)

# the correlation matrix
corr_matrix = np.corrcoef(X_filtered, rowvar=False)

print(corr_matrix)

import pandas as pd


corr_df = pd.DataFrame(corr_matrix, columns=["Radius", "Texture","Perimeter","Area","smoothness","compactness","symmetry","fractal_dimension"],
                        index=["Radius", "Texture","Perimeter","Area","smoothness","compactness","symmetry","fractal_dimension"])


print("Covariance Matrix:")
print(corr_df)
