import gzip
import _pickle as cPickle
import numpy as np


def vectorize(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
    file = gzip.open('mnist.pkl.gz', 'rb')
    imported = cPickle.Unpickler(file=file, encoding='latin1')
    train, val, test = imported.load()
    file.close()
    return train, val, test


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    train_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    train_res = [vectorize(y) for y in tr_d[1]]
    val_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    train = list(zip(train_inputs, train_res))
    val = list(zip(val_inputs, va_d[1]))
    test = list(zip(test_inputs, te_d[1]))
    return train, val, test
