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
