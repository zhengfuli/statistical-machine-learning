import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR

np.random.seed(2018)
n = 600
x = np.random.rand(n)
y = 0.25 + 0.5*x + np.sqrt(0.1)*np.random.randn(n)
idx = np.random.randint(0,600,60)
y[idx] = y[idx] + np.random.randn(60)

xtrain, ytrain = x[:500], y[:500]
xtest, ytest = x[500:], y[500:]

# from hw4 solution: Ridge regression
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

# from hw4 solution: Define function for applying least squares/ridge regression estimate
def lin_reg_appl(X,theta):
    n = X.shape[0]
    d = X.shape[1]
    A = np.zeros((n,d+1))
    A[:,0] = np.ones(n)
    A[:,1:] = X
    y = np.dot(A,theta)
    return y

# directly using package function from sklearn to validate
# reg = linear_model.Ridge(alpha=0.1)
# reg.fit(xtrain.reshape(-1,1), ytrain)
# mse = np.sum((ytest - reg.predict(xtest.reshape(-1,1)))**2) / len(ytest)
# print mse
# print reg.intercept_, reg.coef_[0]

theta_hat = ridge_reg_est(xtrain.reshape(-1,1), ytrain, 0.1)
mse = np.sum((ytest - lin_reg_appl(xtest.reshape(-1,1), theta_hat))**2) / len(ytest)
print mse, theta_hat

reg = linear_model.HuberRegressor(epsilon = 1.2, alpha = 3.6)
reg.fit(xtrain.reshape(-1,1),ytrain)
print reg.intercept_, reg.coef_[0]

svr = SVR(C=0.41, epsilon=0.0405, kernel='linear')
svr.fit(xtrain.reshape(-1,1),ytrain)
print svr.intercept_, svr.coef_[0]
