from sklearn.datasets import load_boston
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

boston = load_boston()
X = boston.data
y = boston.target

X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

X_train = (X_train-X_train.mean(axis=0))/X_train.std(axis=0)
y_train = (y_train-y_train.mean())/y_train.std()
X_test = (X_test-X_train.mean(axis=0))/X_train.std(axis=0)
y_test = (y_test-y_train.mean())/y_train.std()

def leastSquareRegression():
    theta = np.dot(np.linalg.inv(np.dot(np.transpose(X_train),X_train)),
                   np.dot(np.transpose(X_train),y_train))
    mse = np.sum((np.subtract(y_test, np.dot(X_test, theta))*
                  np.subtract(y_test, np.dot(X_test, theta))))/ \
                  X_test.shape[0]
    return mse

def ridgeRegression():
    lambdas = np.linspace(0, 0.004, num=21)
    mses = np.zeros((21,1))
    I = np.eye(X_train.shape[1])

    for i in xrange(len(lambdas)):
        theta = np.dot(np.linalg.inv(np.add(np.dot(np.transpose(X_train), X_train), \
                      (X_train.shape[0]*lambdas[i]*I))), np.dot(np.transpose(X_train), y_train))
        mses[i] = np.sum((np.subtract(y_test,np.dot(X_test,theta))*
                          np.subtract(y_test,np.dot(X_test,theta))))/X_test.shape[0]
    return mses

def LASSO():
    reg = linear_model.Lasso(alpha=0.01)
    pred = reg.fit(X_train, y_train).predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print np.sum(reg.coef_ != 0)
    return mse

if __name__ == '__main__':
    mse = leastSquareRegression()
    print mse
    mses = ridgeRegression()
    print mses
    mse = LASSO()
    print mse