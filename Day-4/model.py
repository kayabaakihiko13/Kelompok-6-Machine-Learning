import numpy as np

class LinearRegression:
    def __init__(self) :
        self.bobot=None
    def fit(self,X,Y):
        X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        self.bobot=np.linalg.inv(X.T @ X)@X.T@Y
        return self

    def predict (self,X) -> np.array:
        X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        return X @ self.bobot
