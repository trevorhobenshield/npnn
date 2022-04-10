import gzip
import hashlib
import requests
from pathlib import Path
import numpy as np
from numpy import r_,c_

def get_data():
    def fn(url):
        if (fp:=Path('/tmp', hashlib.md5(url.encode('utf-8')).hexdigest())).is_file():
            with open(fp,'rb')as f:d=f.read()
        else:
            with open(fp,'wb')as f:f.write(d:=requests.get(url).content)
        return np.frombuffer(gzip.decompress(d),'uint8')
    X_train = fn('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[0x10:]
    y_train = fn('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')[0x08:]
    X_test = fn('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')[0x10:]
    y_test = fn('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')[0x08:]
    X_train, y_train = X_train.reshape(-1, 784, 1), np.eye(10)[y_train][..., np.newaxis]
    X_test, y_test = X_test.reshape(-1, 784, 1), np.eye(10)[y_test][..., np.newaxis]
    return X_train, y_train, X_test, y_test


class MLP:
    def __init__(self, L):
        ep = np.finfo(float).eps  # prevent log(0)
        self.L = L
        self.W = [np.random.normal(0., np.sqrt(2. / self.L[i]), size=(self.L[i + 1], self.L[i])) for i in
                  range(len(self.L) - 1)]  # kaiming/bottou
        self.b = [np.random.randn(i,1) for i in self.L[1:]]
        self.ReLU, self.dReLU = lambda z:np.maximum(0, z), lambda z:np.where(z > 0, 1, 0)
        self.softmax = lambda s:(exps := np.exp(s)) / np.sum(exps)
        self.J,self.dJ = lambda a,y:-np.sum(y * np.log(a + ep)),lambda a,y:a-y
        self.zero_grads = lambda:([np.zeros_like(w) for w in self.W],[np.zeros_like(b) for b in self.b])
        self.Z,self.A = [],[]  # cache forward computations

    def SGD(self, train, epochs=10, batch_size=16, lr=0.001, reg=1e-3):
        train_size = len(train)
        for epoch in r_[:epochs]:
            total_loss = 0.
            print(f'{epoch = }')
            train = np.random.permutation(train)
            for i, batch in enumerate(train[n:n + batch_size] for n in range(0, train_size, batch_size)):
                dW,db = self.zero_grads()
                loss = 0.
                for j,(X,y) in enumerate(batch):
                    loss += self.forward(X,y)
                    d_dW,d_db = self.backward(y)
                    dW = np.sum([dW,d_dW], axis=0)
                    db = np.sum([db,d_db], axis=0)
                self.W = [w-(lr*reg*w/train_size + lr*dw/batch_size) for w, dw in zip(self.W, dW)]
                self.b -= lr*db
                total_loss += loss
            print(f'{total_loss = }')

    def forward(self,X,y):
        self.A,self.Z = [X],[]
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            Z = W @ self.A[i] + b
            A = self.ReLU(Z)
            self.Z.append(Z)
            self.A.append(A)
        self.A[-1] = self.softmax(self.A[-1])
        return self.J(self.A[-1],y)

    def backward(self,y):
        dW,db = self.zero_grads()
        d = self.dJ(self.A[-1],y)
        db[-1] = d
        dW[-1] = d @ self.A[-2].T
        for l in -np.arange(1,len(self.L)-1):
            d = (self.W[l].T @ d) * self.dReLU(self.Z[l-1])
            db[l-1] = d
            dW[l-1] = d @ self.A[l-2].T
        return dW,db


def main():
    np.random.seed(69)
    X_train, y_train = get_data()[:2]
    train = list(zip(X_train, y_train))
    net = MLP([784, *[64, 32, 16], 10])
    net.SGD(train=train,
            epochs=10,
            batch_size=32,
            lr=0.0001)


if __name__ == '__main__':
    main()
