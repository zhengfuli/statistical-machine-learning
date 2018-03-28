import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from math import exp
import time

# the logistic function
def logistic_func(theta, x):
    t = x.dot(theta)
    g = np.zeros(t.shape)
    # split into positive and negative to improve stability
    g[t >= 0.0] = 1.0 / (1.0 + np.exp(-t[t >= 0.0]))
    g[t < 0.0] = np.exp(t[t < 0.0]) / (np.exp(t[t < 0.0]) + 1.0)
    return g

# function to compute log-likelihood
def neg_log_like(theta, x, y):
    g = logistic_func(theta, x)
    return -sum(np.log(g[y > 0.5])) - sum(np.log(1 - g[y < 0.5]))

# function to compute the gradient of the negative log-likelihood
def log_grad(theta, x, y):
    g = logistic_func(theta, x)
    return -x.T.dot(y - g)

# implementation of gradient descent for logistic regression
# def grad_desc(theta, x, y, alpha, tol, maxiter):
#     nll_vec = []
#     nll_vec.append(neg_log_like(theta, x, y))
#     nll_delta = 2.0 * tol
#     iter = 0
#     while (nll_delta > tol) and (iter < maxiter):
#         theta -= alpha * log_grad(theta, x, y)
#         nll_vec.append(neg_log_like(theta, x, y))
#         nll_delta = nll_vec[-2] - nll_vec[-1]
#         iter += 1
#     print iter, nll_delta
#     return theta, np.array(nll_vec)

def Newt_Raph(theta,x):
    g = logistic_func(theta,x)
    H = np.zeros((3,3))
    for i in range(0,len(x)):
        x0 = np.array([x[i]])
        xT = np.array([x[i]]).T
        H = H+np.dot(xT,x0)*g[i]*(1-g[i])
    H = np.linalg.inv(H)
    return H

# Newton method
def grad_desc(theta, x, y, tol, maxiter):
    # nll_vec = []
    # nll_vec.append(neg_log_like(theta, x, y))
    # nll_delta = 2.0 * tol
    # iter = 0
    # while abs(nll_delta) > tol and iter < maxiter:
    #     g = logistic_func(theta, x)
    #     h = x.T.dot(np.diag(g*(1-g))).dot(x)
    #     theta -= np.linalg.inv(h).dot(log_grad(theta, x, y))
    #     nll_vec.append(neg_log_like(theta, x, y))
    #     nll_delta = nll_vec[-2] - nll_vec[-1]
    #     iter += 1
    # print iter, nll_delta
    # return theta, np.array(nll_vec)

    nll_vec = []
    nll_vec.append(neg_log_like(theta, x, y))
    nll_delta = 2.0*tol
    iter = 0
    while (abs(nll_delta) > tol) and (iter < maxiter):
        alpha = Newt_Raph(theta,x)
        theta = theta - (alpha.dot(log_grad(theta, x, y)))
        nll_vec.append(neg_log_like(theta, x, y))
        nll_delta = nll_vec[-2]-nll_vec[-1]
        iter += 1
    print iter, nll_delta
    return theta, np.array(nll_vec)

# stochastic gradient descent
# def grad_desc(theta, x, y, alpha, tol, maxiter):
#     nll_vec = []
#     nll_vec.append(neg_log_like(theta, x, y))
#     nll_delta = 2.0 * tol
#     iter = 0
#     while abs(nll_delta) > tol and iter < maxiter:
#         rand = np.random.randint(0, len(x))
#         theta -= alpha * log_grad(theta, x[rand], y[rand])
#         nll_vec.append(neg_log_like(theta, x, y))
#         nll_delta = nll_vec[-2] - nll_vec[-1]
#         iter += 1
#     print iter, nll_delta
#     return theta, np.array(nll_vec)

# function to compute output of LR classifier
def lr_predict(theta, x):
    # form Xtilde for prediction
    shape = x.shape
    Xtilde = np.zeros((shape[0], shape[1] + 1))
    Xtilde[:, 0] = np.ones(shape[0])
    Xtilde[:, 1:] = x
    return logistic_func(theta, Xtilde)

## Generate dataset
np.random.seed(2017)  # Set random seed so results are repeatable
x, y = datasets.make_blobs(n_samples=100000, n_features=2, centers=2, cluster_std=6.0)

## build classifier
# form Xtilde
shape = x.shape
xtilde = np.zeros((shape[0], shape[1] + 1))
xtilde[:, 0] = np.ones(shape[0])
xtilde[:, 1:] = x

# Initialize theta to zero
theta = np.zeros(shape[1] + 1)

# Run gradient descent
alpha = 0.000001
tol = 1e-3
maxiter = 10000
start_time = time.time()
theta, cost = grad_desc(theta, xtilde, y, tol, maxiter)
print time.time()-start_time
# ## Plot the decision boundary.
# # Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
# h = .02  # step size in the mesh
# x_delta = (x[:, 0].max() - x[:, 0].min()) * 0.05  # add 5% white space to border
# y_delta = (x[:, 1].max() - x[:, 1].min()) * 0.05
# x_min, x_max = x[:, 0].min() - x_delta, x[:, 0].max() + x_delta
# y_min, y_max = x[:, 1].min() - y_delta, x[:, 1].max() + y_delta
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = lr_predict(theta, np.c_[xx.ravel(), yy.ravel()])
#
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
# ## Plot the training points
# plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold)
#
# ## Show the plot
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("Logistic regression classifier")
# plt.show()