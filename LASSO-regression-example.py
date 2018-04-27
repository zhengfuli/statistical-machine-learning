# from sklearn.datasets import load_boston
# from sklearn import linear_model
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# import numpy as np
#
# boston = load_boston()
# X = boston.data
# y = boston.target
#
# X_train, y_train = X[:400], y[:400]
# X_test, y_test = X[400:], y[400:]
#
# X_train = (X_train-X_train.mean(axis=0))/X_train.std(axis=0)
# y_train = (y_train-y_train.mean())/y_train.std()
# X_test = (X_test-X_train.mean(axis=0))/X_train.std(axis=0)
# y_test = (y_test-y_train.mean())/y_train.std()
#
# def leastSquareRegression():
#     theta = np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),
#                    np.dot(np.transpose(X_train),y_train))
#     mse = np.sum((np.subtract(y_test, np.dot(X_test, theta))*
#                   np.subtract(y_test, np.dot(X_test, theta))))/ \
#                   X_test.shape[0]
#     return mse
#
# def ridgeRegression():
#     lambdas = np.linspace(0, 0.004, num=21)
#     mses = np.zeros((21,1))
#     I = np.eye(X_train.shape[1])
#
#     for i in xrange(len(lambdas)):
#         theta = np.dot(np.linalg.inv(np.add(np.dot(np.transpose(X_train), X_train), \
#                       (X_train.shape[0]*lambdas[i]*I))), np.dot(np.transpose(X_train), y_train))
#         mses[i] = np.sum((np.subtract(y_test,np.dot(X_test,theta))*
#                           np.subtract(y_test,np.dot(X_test,theta))))/X_test.shape[0]
#     return mses
#
# def LASSO():
#     reg = linear_model.Lasso(alpha=0.01)
#     pred = reg.fit(X_train, y_train).predict(X_test)
#     mse = mean_squared_error(y_test, pred)
#     print np.sum(reg.coef_ != 0)
#     return mse
#
# if __name__ == '__main__':
#     mse = leastSquareRegression()
#     print mse
#     mses = ridgeRegression()
#     print mses
#     mse = LASSO()
#     print mse

import numpy as np

# Define function for computing least squares estimate
def least_squares_est(X,y):
    n = X.shape[0]
    d = X.shape[1]
    A = np.zeros((n,d+1))
    A[:,0] = np.ones(n)
    A[:,1:] = X
    theta = np.dot(np.linalg.inv(np.dot(A.T,A)),np.dot(A.T,y))
    return theta

# Define function for computing ridge regression estimate
def ridge_reg_est(X,y,lam):
    n = X.shape[0]
    d = X.shape[1]
    A = np.zeros((n,d+1))
    A[:,0] = np.ones(n)
    A[:,1:] = X
    Gam = np.zeros((d+1,d+1))
    np.fill_diagonal(Gam,lam)
    Gam[0,0] = 0.0
    theta = np.dot(np.linalg.inv(np.dot(A.T,A)+Gam),np.dot(A.T,y))
    return theta

# Define function for applying least squares/ridge regression estimate
def lin_reg_appl(X,theta):
    n = X.shape[0]
    d = X.shape[1]
    A = np.zeros((n,d+1))
    A[:,0] = np.ones(n)
    A[:,1:] = X
    y = np.dot(A,theta)
    return y

## Import data
from sklearn.datasets import load_boston
boston = load_boston()

X = boston.data
y = boston.target

## Split into training and testing data
Xtrain = X[0:400,:]
ytrain = y[0:400]
Xtest = X[400:,:]
ytest = y[400:]

## Standardize training/testing data
mu = np.mean(Xtrain,axis=0)
sigma = np.std(Xtrain,axis=0)
Xtrain_std = (Xtrain-mu)/sigma
Xtest_std = (Xtest-mu)/sigma

## Least squares
theta_hat = least_squares_est(Xtrain_std,ytrain)
yhat = lin_reg_appl(Xtest_std,theta_hat)
mse = np.sum((ytest - yhat)**2)/len(ytest)

print 'Least squares MSE: ' + str(mse)

## Ridge regression

# Select optimal lambda using holdout set
# lambda_vec = np.logspace(2,3,11)
# for lam in lambda_vec:
#     theta_hat = ridge_reg_est(Xtrain_std[0:300,:],ytrain[0:300],lam)
#     mse = np.sum((ytrain[300:] - lin_reg_appl(Xtrain_std[300:,:],theta_hat))**2)/100
#     print 'Ridge regression MSE for lambda = ' + str(lam) + ' : ' + str(mse)

lam = 350
theta_hat = ridge_reg_est(Xtrain_std,ytrain,lam)
mse = np.sum((ytest - lin_reg_appl(Xtest_std,theta_hat))**2)/len(ytest)
print 'Ridge regression MSE : ' + str(mse)

## Lasso
from sklearn import linear_model
# Select optimal alpha using holdout set
#alpha_vec = np.logspace(-1,1,21)
#for alph in alpha_vec:
# reg = linear_model.Lasso(alpha=alph,normalize=False)
# reg.fit(Xtrain_std[0:250,:],ytrain[0:250])
# mse = np.sum((ytrain[250:] - reg.predict(Xtrain_std[250:,:]))**2)/150
# print 'LASSO MSE for alpha = ' + str(alph) + ': ' + str(mse)

reg = linear_model.Lasso(alpha=0.5)
reg.fit(Xtrain_std,ytrain)
mse = np.sum((ytest - reg.predict(Xtest_std))**2)/len(ytest)
print 'LASSO MSE : ' + str(mse)
print 'Number of nonzeros : ' + str(np.count_nonzero(reg.coef_))