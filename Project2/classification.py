import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../Data/Prostate_Cancer.csv', index_col="id")
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

X = df.to_numpy() # df as a numpy array
# Field selected for regression:
target_column = "diagnosis_result"


y = df[target_column]
X = df[continuous_cols]

X = X.to_numpy()
y = y.to_numpy()
N, M = X.shape

def logistic_regresion():
    # Create crossvalidation partition for evaluation
    # using stratification and 95 pct. split between training and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y)
    # Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
    # effect of regularization? How does differetn runs of  test_size=.99 compare 
    # to eachother?

    # Fit regularized logistic regression model to training data to predict 
    # the type of wine
    lambda_interval = np.logspace(-8, 2, 50)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    for k in range(0, len(lambda_interval)):
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
        
        mdl.fit(X_train, y_train)

        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

        w_est = mdl.coef_[0] 
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]

    plt.figure(figsize=(8,8))
    # plt.plot(np.log10(lambda_interval), train_error_rate*100)
    # plt.plot(np.log10(lambda_interval), test_error_rate*100)
    # plt.plot(np.log10(opt_lambda), min_error*100, 'o')
    plt.semilogx(lambda_interval, train_error_rate)
    plt.semilogx(lambda_interval, test_error_rate)
    plt.semilogx(opt_lambda, min_error, 'o')
    plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
    plt.ylim([0, 4])
    plt.grid()
    plt.show()    

    plt.figure(figsize=(8,8))
    plt.semilogx(lambda_interval, coefficient_norm,'k')
    plt.ylabel('L2 Norm')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.title('Parameter vector L2 norm')
    plt.grid()
    plt.show()    

def baseline():
    # Load and preprocess the data (as shown in the original code)
    # Assuming df and y are prepared as before

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)

    # Compute the largest class in the training data
    largest_class = np.argmax(np.bincount(y_train))

    # Baseline model predicting the test data as the largest class from training data
    baseline_model = DummyClassifier(strategy='constant', constant=largest_class)
    baseline_model.fit(X_train, y_train)  # Fitting the baseline model

    # Generate predictions for the test data using the baseline model
    baseline_predictions = baseline_model.predict(X_test)

    # Plotting the results
    plt.figure(figsize=(8, 6))

    plt.scatter(range(len(y_test)), y_test, label='True Values', marker='o')
    plt.scatter(range(len(baseline_predictions)), baseline_predictions, label='Predicted Values', marker='x')
    plt.legend()
    plt.xlabel('Data Index')
    plt.ylabel('Class')
    plt.title('Baseline Model Predictions vs True Values')
    plt.grid()

    plt.show()


def KNN():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, stratify=y)

    # Define the range of neighbors (k) for KNN
    k_values = range(1, 20)  # Example range from 1 to 20

    train_error_rate = np.zeros(len(k_values))
    test_error_rate = np.zeros(len(k_values))

    # Fit KNN models with different values of k
    for i, k in enumerate(k_values):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        y_train_est = knn.predict(X_train)
        y_test_est = knn.predict(X_test)

        # Calculate error rates
        train_error_rate[i] = np.mean(y_train_est != y_train)
        test_error_rate[i] = np.mean(y_test_est != y_test)

    # Determine the optimal k value based on the minimum test error
    optimal_k = k_values[np.argmin(test_error_rate)]

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, train_error_rate, label='Training Error Rate')
    plt.plot(k_values, test_error_rate, label='Test Error Rate')
    plt.scatter(optimal_k, min(test_error_rate), color='red', label='Optimal K')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Error Rate')
    plt.title('KNN Model: Error Rate vs. Number of Neighbors')
    plt.legend()
    plt.grid()
    plt.show()


KNN()
