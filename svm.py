from getEmbeddings import getEmbeddings
from getEmbeddings2 import getEmbeddings2
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os
import pickle

def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()

# Read the data
if not os.path.isfile('./xtr.npy') or \
    not os.path.isfile('./xte.npy') or \
    not os.path.isfile('./ytr.npy') or \
    not os.path.isfile('./yte.npy'):
    xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
    np.save('./xtr', xtr)
    np.save('./xte', xte)
    np.save('./ytr', ytr)
    np.save('./yte', yte)

if not os.path.isfile('./xtr2.npy') or \
    not os.path.isfile('./xte2.npy') or \
    not os.path.isfile('./ytr2.npy') or \
    not os.path.isfile('./yte2.npy'):
    xtr2,xte2,ytr2,yte2 = getEmbeddings2("datasets/train.csv")
    np.save('./xtr2', xtr2)
    np.save('./xte2', xte2)
    np.save('./ytr2', ytr2)
    np.save('./yte2', yte2)

xtr = np.load('./xtr.npy')
xte = np.load('./xte.npy')
ytr = np.load('./ytr.npy')
yte = np.load('./yte.npy')

xtr2 = np.load('./xtr2.npy')
xte2 = np.load('./xte2.npy')
ytr2 = np.load('./ytr2.npy')
yte2 = np.load('./yte2.npy')

# SVM for classification
clf = SVC()
clf.fit(xtr, ytr)
y_pred = clf.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy of Priority = " + format((m-n)/m*100, '.2f') + "%")

clf2 = SVC()
clf2.fit(xtr2, ytr2)
y_pred2 = clf2.predict(xte2)
m = yte2.shape[0]
n = (yte2 != y_pred2).sum()
print("Accuracy of Tags = " + format((m-n)/m*100, '.2f') + "%")

filename = 'priority.sav'
pickle.dump(clf, open(filename, 'wb'))

filename1 = 'tags.sav'
pickle.dump(clf2, open(filename1, 'wb'))

# Draw the confusion matrix
# plot_cmat(yte, y_pred)
# plot_cmat(yte2, y_pred2)
