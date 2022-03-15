import numpy as np

class MLP:
    def __init__(self, L):
        self.L = L
        self.W = [np.random.normal(0., np.sqrt(2. / self.L[i]), size=(self.L[i + 1], self.L[i])) for i in range(len(self.L) - 1)]
        self.b = [np.random.randn(y, 1) for y in self.L[1:]]
        self.f, self.df = lambda x: 1. / (1. + np.exp(-x)), lambda x: self.f(x) * (1. - self.f(x))
        self.C, self.dC = lambda a, y: np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a))),lambda a, y: a - y
        self.Z, self.A = ..., ...
        self.zero_grads = lambda: ([np.zeros_like(w) for w in self.W], [np.zeros_like(b) for b in self.b])

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
                loss += .5 * (reg / len(train)) * sum(np.linalg.norm(w) ** 2 for w in self.W)
                if i % 200 == 199: print(f'{loss = }')

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
        return self.C(self.A[-1], y)

    def backward(self, y):
        gW, gb = self.zero_grads()
        d = self.dC(self.A[-1], y)
        gb[-1] = d
        gW[-1] = d @ self.A[-2].T
        l = -1
        while l > -len(self.L) + 1:
            d = self.W[l].T @ d * self.df(self.Z[l - 1])
            gb[l - 1] = d
            gW[l - 1] = d @ self.A[l - 2].T
            l -= 1
        return gW, gb
