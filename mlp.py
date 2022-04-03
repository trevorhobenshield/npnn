import gzip
import hashlib
import requests
from pathlib import Path
import numpy as np


def get_data():
    def fn(url):
        fp = Path('/tmp', hashlib.md5(url.encode('utf-8')).hexdigest())
        if fp.is_file():
            with open(fp, 'rb') as fr:
                dat = fr.read()
        else:
            with open(fp, 'wb') as fw:
                fw.write(dat := requests.get(url).content)
        return np.frombuffer(gzip.decompress(dat), 'uint8')

    # img dim offset = [0xB]
    X_train = fn('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[0x10:]
    y_train = fn('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')[0x08:]
    X_test = fn('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')[0x10:]
    y_test = fn('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')[0x08:]

    X_train = X_train.reshape(-1, 784, 1)
    X_test = X_test.reshape(-1, 784, 1)
    y_train = np.eye(10)[y_train][..., np.newaxis]
    y_test = np.eye(10)[y_test][..., np.newaxis]

    return X_train, y_train, X_test, y_test


class MLP:
    def __init__(self, L):
        self.L = L
        self.W = [np.random.normal(0., np.sqrt(2. / self.L[i]), size=(self.L[i + 1], self.L[i])) for i in
                  range(len(self.L) - 1)]
        self.b = [np.random.randn(y, 1) for y in self.L[1:]]
        self.Z, self.A = ..., ...
        self.zero_grads = lambda:([np.zeros_like(w) for w in self.W], [np.zeros_like(b) for b in self.b])

    def f(self, z):
        """ReLU"""
        return np.maximum(0, z)

    def df(self, z):
        """Derivative of ReLU"""
        return np.where(z > 0, 1, 0)

    def softmax(self, s):
        """Softmax (unstable)"""
        exps = np.exp(s)
        probs = exps / np.sum(exps)
        return probs

    def J(self, y_hat, y):
        """Cross-Entropy Loss function"""
        eps = np.finfo(float).eps  # prevent computing log(0)
        ce = -np.sum(y * np.log(y_hat + eps))
        return ce

    def dJ(self, y_hat, y):
        """Derivative of Cross-Entropy Loss function"""
        return y_hat - y

    def forward(self, X, y):
        self.A = [X]
        self.Z = []
        # p = 0.5
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = W @ self.A[i] + b
            A = self.f(Z)
            # A *= (np.random.rand(*A.shape)<p)/p ## dropout
            self.Z.append(Z)
            self.A.append(A)
        self.A[-1] = self.softmax(self.A[-1])
        return self.J(self.A[-1], y)

    def backward(self, y):
        gW, gb = self.zero_grads()
        d = self.dJ(self.A[-1], y)
        gb[-1] = d
        gW[-1] = d @ self.A[-2].T
        l = -1
        while l > -len(self.L) + 1:
            d = (self.W[l].T @ d) * self.df(self.Z[l - 1])
            gb[l - 1] = d
            gW[l - 1] = d @ self.A[l - 2].T
            l -= 1
        return gW, gb

    def SGD(self, train, epochs=3, batch_size=32, lr=0.01, reg=0.5):
        m = len(train)
        for epoch in range(epochs):
            train = np.random.permutation(train)
            print(f'{epoch = }')
            for i, mini_batch in enumerate(train[n:n + batch_size] for n in range(0, m, batch_size)):
                gW, gb = self.zero_grads()
                loss = 0.
                for j, (X, y) in enumerate(mini_batch):
                    loss += self.forward(X, y)
                    dgW, dgb = self.backward(y)
                    gW = np.sum([gW, dgW], axis=0)
                    gb = np.sum([gb, dgb], axis=0)
                self.W = [(1 - lr * (reg / m)) * W - (lr / len(mini_batch)) * gW for W, gW in zip(self.W, gW)]
                self.b = [b - ((lr * gb) / len(mini_batch)) for b, gb in zip(self.b, gb)]
                loss += .5 * (reg / len(train)) * sum(np.linalg.norm(w)**2 for w in self.W)
                if i % 200 == 199: print(f'{loss = }')


def main():
    np.random.seed(69)

    X_train, y_train = get_data()[:2]
    train = list(zip(X_train, y_train))
    net = MLP([784, *[64, 32, 16], 10])
    net.SGD(train=train,
            epochs=3,
            batch_size=16,
            lr=0.0001,
            reg=0.2)


if __name__ == '__main__':
    main()
