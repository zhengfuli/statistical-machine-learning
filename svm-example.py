from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import svm

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

# dividing data sets
X4, y4 = X[y==4,:], y[y==4]
X9, y9 = X[y==9,:], y[y==9]

Xfit = np.concatenate((X4[0:3000],X9[0:3000]))
yfit = np.concatenate((y4[0:3000],y9[0:3000]))
Xholdout = np.concatenate((X4[3000:4000],X9[3000:4000]))
yholdout = np.concatenate((y4[3000:4000],y9[3000:4000]))
# When getting test error, use the whole training set to fit
Xtrain = np.concatenate((X4[0:4000],X9[0:4000]))
ytrain = np.concatenate((y4[0:4000],y9[0:4000]))
Xtest = np.concatenate((X4[4000:],X9[4000:]))
ytest = np.concatenate((y4[4000:],y9[4000:]))

# (1) C = 0.001, Pe = 0.0265(holdout) for degree = 1
#     C = 0.000001, Pe = 0.0095(holdout) for degree = 2
clf1 = svm.SVC(0.001, kernel='poly', degree=1)
clf1.fit(Xtrain, ytrain)
print("Test Error = %f, SV Num = %f"
      %((1 - clf1.score(Xtest, ytest)), len(clf1.support_vectors_)))

clf2 = svm.SVC(0.000001, kernel='poly', degree=2)
clf2.fit(Xtrain, ytrain)
print("Test Error = %f, SV Num = %f"
      %((1 - clf2.score(Xtest, ytest)), len(clf2.support_vectors_)))
# (2) C = 10, Gamma = 10^-6, Pe = 0.01(holdout)
clf3 = svm.SVC(10, kernel='rbf', gamma=10**(-6))
clf3.fit(Xtrain, ytrain)
print("Test Error = %f, SV Num = %f,%s"
      %((1 - clf3.score(Xtest, ytest)), len(clf3.support_vectors_),clf3.support_vectors_.shape))

sv1, sv2, sv3 = clf1.support_vectors_, \
                clf2.support_vectors_, \
                clf3.support_vectors_
svlabel1, svlabel2, svlabel3 = ytrain[clf1.support_], \
                               ytrain[clf2.support_], \
                               ytrain[clf3.support_]
distance1, distance2, distance3 = clf1.decision_function(sv1), \
                                  clf2.decision_function(sv2), \
                                  clf3.decision_function(sv3)

sv = [sv1, sv2, sv3]
svlabel = [svlabel1, svlabel2, svlabel3]
svd = [distance1, distance2, distance3]
res = [[],[],[]]
dict = [{},{},{}]

for i in range(len(svd)):
    for j in xrange(len(svd[i])):
        dict[i][j]= abs(svd[i][j])

    temp = sorted(dict[i].items(),key=lambda item:item[1])[0:16]

    for k in range(len(temp)):
        res[i].append(temp[k][0])

f1, axarr1 = plt.subplots(4, 4)
f2, axarr2 = plt.subplots(4, 4)
f3, axarr3 = plt.subplots(4, 4)
axarr = [axarr1, axarr2, axarr3]

for i in range(len(res)):
    for j in range(4):
        for k in range(4):
            axarr[i][j, k].imshow(sv[i][res[i][k+4*j]].reshape((28,28)), cmap='gray')
            axarr[i][j, k].axis('off')
            axarr[i][j, k].set_title('{label}'.format(label=int(svlabel[i][res[i][k+4*j]])))
plt.show()