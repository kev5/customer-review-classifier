import pickle
import csv
import numpy as np
from io import StringIO
import pandas as pd
from numpy import genfromtxt
import string
import re
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords


def textClean(text):
    """
    Get rid of the non-letter and non-number characters
    """
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences


def getEmbeddings(path,vector_dimension=300):
    """
    Generate Doc2Vec training and testing data
    """
    data = pd.read_csv(path)

    for i in range(len(data)):
        data.loc[i, 'Message'] = cleanup(data.loc[i,'Message'])

    x = constructLabeledSentences(data['Message'])
    
    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=20,
                         seed=1)
    text_model.build_vocab(x)
    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_test_arrays = np.zeros((test_size, vector_dimension))
    test_labels = np.zeros(test_size)

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        j = j + 1

    return text_test_arrays

# load the model from disk
# filename = 'classification/priority.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score("Viasat is awesome", "0")
# print(result)

# filename = 'classification/tags.sav'
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score("Viasat is awesome", "Praise")
# print(result)

text = 'this router sucks'

s = StringIO(text)
with open('temp.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Message"])
    for line in s:
        f.write(line)


#####################if request == "tags":
filename = 'tags.sav'
X_test = getEmbeddings("temp.csv")
loaded_model = pickle.load(open(filename, 'rb'))
prediction = loaded_model.predict(X_test)

if prediction == 0:
    print("Inquiry")
if prediction == 1:
    print("Account")
if prediction == 2:
    print("Service")
if prediction == 3:
    print("Praise")
if prediction == 4:
    print("DNR")
if prediction == 5:
    print("Technical")


#####################if request == "priority":
filename = 'priority.sav'
X_test = getEmbeddings("temp.csv")
loaded_model = pickle.load(open(filename, 'rb'))
prediction = loaded_model.predict(X_test)

if prediction == 0:
    print("Low")
if prediction == 1:
    print("High")
