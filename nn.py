import numpy as np
from icecream import ic,argumentToString
from numpy import nan_to_num
from numpy.random import permutation,normal

@argumentToString.register(np.ndarray)
def _(a):return f"{a}\t{a.shape = }"

class MLP:
    def __init__(self,L):
        self.L = L
        self.W = [normal(0.,np.sqrt(2./self.L[i]),size=(self.L[i+1],self.L[i]))for i in range(len(self.L)-1)]
        self.b = [np.random.randn(y,1)for y in self.L[1:]]
        self.f,self.df = lambda x:1./(1.+np.exp(-x)),lambda x:self.f(x)*(1.-self.f(x))
        self.C,self.dC = lambda a,y:np.sum(nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))),lambda a,y:a-y
        self.Z,self.A = ...,...
        self.zero_grads = lambda:([np.zeros_like(w)for w in self.W],[np.zeros_like(b)for b in self.b])

    def SGD(self,train,epochs=3,sz=32,lr=0.01,reg=0.5):
        """ Mini-Batch Stochastic Gradient Descent """
        m = len(train)
        for epoch in range(epochs):
            train = permutation(train)
            ic(epoch)
            for i,mb in enumerate(train[n:n+sz]for n in range(0,m,sz)):
                gW,gb = self.zero_grads()
                mb_loss = 0.
                for j,(X,y) in enumerate(mb):
                    mb_loss += self.forward(X,y)/sz
                    dgW,dgb = self.backward(y)
                    gW = np.sum([gW,dgW],axis=0)
                    gb = np.sum([gb,dgb],axis=0)
                self.W = [(1-lr*(reg/m))*W-(lr/len(mb))*gW for W,gW in zip(self.W,gW)]
                self.b = [b-((lr*gb)/len(mb))for b,gb in zip(self.b,gb)]
                mb_loss += .5*(reg/len(train))*sum(np.linalg.norm(w)**2 for w in self.W)
                if i%200==199:ic(mb_loss)

    def forward(self,X,y):
        self.A = [X]
        self.Z = []
        # p = 0.5
        for i,(W,b) in enumerate(zip(self.W,self.b)):
            Z = W @ self.A[i]+b
            A = self.f(Z)
            # A *= (np.random.rand(*A.shape)<p)/p ## dropout
            self.Z.append(Z)
            self.A.append(A)
        return self.C(self.A[-1],y)

    def backward(self,y):
        gW,gb = self.zero_grads()
        d = self.dC(self.A[-1],y)
        gb[-1] = d
        gW[-1] = d @ self.A[-2].T
        for l in range(2,len(self.L)):
            d = self.W[-l+1].T @ d*self.df(self.Z[-l])
            gb[-l] = d
            gW[-l] = d @ self.A[-l-1].T
        return gW,gb
