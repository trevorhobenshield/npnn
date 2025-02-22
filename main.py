from pathlib import Path
import numpy as np


def get_data(data_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def read(fname: str) -> np.ndarray:
        return np.frombuffer((Path(data_path) / fname).read_bytes(), "uint8")

    n_labels = 10
    dim = -1, 784, 1
    X_train, y_train = read("train-images-idx3-ubyte")[0x10:], read("train-labels-idx1-ubyte")[0x08:]
    X_test, y_test = read("t10k-images-idx3-ubyte")[0x10:], read("t10k-labels-idx1-ubyte")[0x08:]
    X_train, y_train = X_train.reshape(*dim), np.eye(n_labels)[y_train][..., np.newaxis]
    X_test, y_test = X_test.reshape(*dim), np.eye(n_labels)[y_test][..., np.newaxis]
    return X_train, y_train, X_test, y_test


class MLP:
    def __init__(self, L: list[int]):
        self.L = L
        self.W = [np.random.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in)) for n_in, n_out in zip(L[:-1], L[1:])]
        self.b = [np.random.randn(n_out, 1) for n_out in L[1:]]
        self.ReLU = lambda z: np.maximum(0, z)
        self.dReLU = lambda z: z > 0
        self.softmax = lambda s: np.exp(s) / np.sum(np.exp(s), axis=0)
        self.loss = lambda a, y: -np.sum(y * np.log(a + 1e-10))
        self.d_loss = lambda a, y: a - y

    def SGD(self, train: list[tuple[np.ndarray, np.ndarray]], epochs: int = 10, batch_size: int = 32, lr: float = 0.001, reg: float = 1e-3):
        train_size = len(train)
        for epoch in range(epochs):
            indices = np.random.permutation(train_size)
            total_loss = 0.0
            for i in range(0, train_size, batch_size):
                batch = [train[idx] for idx in indices[i : i + batch_size]]
                dW = [np.zeros_like(w) for w in self.W]
                db = [np.zeros_like(b) for b in self.b]
                loss = 0.0
                for X, y in batch:
                    # forward
                    A = [X]
                    Z = []
                    for w, b in zip(self.W, self.b):
                        Z.append(w @ A[-1] + b)
                        A.append(self.ReLU(Z[-1]))
                    A[-1] = self.softmax(Z[-1])
                    loss += self.loss(A[-1], y)

                    delta = self.d_loss(A[-1], y)
                    db[-1] += delta
                    dW[-1] += delta @ A[-2].T
                    for l in reversed(range(len(self.W) - 1)):
                        delta = (self.W[l + 1].T @ delta) * self.dReLU(Z[l])
                        db[l] += delta
                        dW[l] += delta @ A[l].T

                for l in range(len(self.W)):
                    self.W[l] -= lr * (dW[l] / batch_size + reg * self.W[l] / train_size)
                    self.b[l] -= lr * db[l] / batch_size

                total_loss += loss / batch_size
            print(f"{epoch = }\t{loss = }\t{total_loss = }")


def main(data_path: str | Path):
    np.random.seed(69)
    X_train, y_train = get_data(data_path)[:2]
    train = list(zip(X_train, y_train))
    net = MLP([784, *[64, 32, 16], 10])
    net.SGD(train=train, epochs=10, batch_size=32, lr=0.0001)


if __name__ == "__main__":
    main(data_path=Path(__file__).parent / "mnist")
