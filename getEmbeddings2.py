import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder


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


def getEmbeddings2(path,vector_dimension=6):
    """
    Generate Doc2Vec training and testing data
    """
    data = pd.read_csv(path)

    # missing_rows = []
    # for i in range(len(data)):
    #     if data.loc[i, 'Message'] != data.loc[i, 'Message']:
    #         missing_rows.append(i)
    # data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'Message'] = cleanup(data.loc[i,'Message'])

    x = constructLabeledSentences(data['Message'])
    y1 = data['Tags'].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y1)

    text_model = Doc2Vec(min_count=1, window=5, vector_size=vector_dimension, sample=1e-4, negative=5, workers=7, epochs=20,
                         seed=1)
    text_model.build_vocab(x)
    text_model.train(x, total_examples=text_model.corpus_count, epochs=text_model.iter)

    train_size = int(0.8 * len(x))
    test_size = len(x) - train_size

    text_train_arrays = np.zeros((train_size, vector_dimension))
    text_test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    for i in range(train_size):
        text_train_arrays[i] = text_model.docvecs['Text_' + str(i)]
        train_labels[i] = y[i]

    j = 0
    for i in range(train_size, train_size + test_size):
        text_test_arrays[j] = text_model.docvecs['Text_' + str(i)]
        test_labels[j] = y[i]
        j = j + 1

    return text_train_arrays, text_test_arrays, train_labels, test_labels


def clean_data():
    """
    Generate processed string
    """
    path = 'datasets/train.csv'
    vector_dimension=300

    data = pd.read_csv(path, error_bad_lines=False)

    missing_rows = []
    for i in range(len(data)):
        if data.loc[i, 'Message'] != data.loc[i, 'Message']:
            missing_rows.append(i)
    data = data.drop(missing_rows).reset_index().drop(['index','id'],axis=1)

    for i in range(len(data)):
        data.loc[i, 'Message'] = cleanup(data.loc[i,'Message'])

    data = data.sample(frac=1).reset_index(drop=True)

    x = data.loc[:,'Message'].values
    y = data.loc[:,'Tags'].values

    train_size = int(0.8 * len(y))
    test_size = len(x) - train_size

    xtr2 = x[:train_size]
    xte2 = x[train_size:]
    ytr2 = y[:train_size]
    yte2 = y[train_size:]

    np.save('xtr_shuffled2.npy',xtr2)
    np.save('xte_shuffled2.npy',xte2)
    np.save('ytr_shuffled2.npy',ytr2)
    np.save('yte_shuffled2.npy',yte2)
