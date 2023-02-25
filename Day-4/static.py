import numpy as np
from collections import Counter
import math

def r2_score(y_actual,y_pred):
    """
    
    https://medium.com/@himanshuit3036/simple-linear-regression-from-scratch-using-numpy-910a3f838729
    
    example:
    ---
    >>> r2_score(Y,y_pred)
    0.8909551175165973
    
    """

    rss=np.sum(np.square(y_actual-y_pred))
    tss=np.sum(np.square(y_actual-np.mean(y_actual)))

    return round(1-(rss/tss),2)

class LinearRegression:
    def __init__(self):
        self.bobot = None
    def fit(self,X,Y):
        X = np.concatenate((np.ones((X.shape[0],1)) , X) , axis=1)
        self.bobot = np.linalg.inv(X.T @ X) @ X.T @ Y
        return self
    
    def fit_transforms(self,X,Y):
        X = np.concatenate((np.ones((X.shape[0],1)) , X) , axis=1)
        self.bobot = np.linalg.inv(X.T @ X) @ X.T @ Y
        return self.bobot
    
    def predict(self,X) -> np.array:
        X = np.concatenate((np.ones((X.shape[0] , 1)) , X) , axis=1)
        return X @ self.bobot

class Knn:
    def __init__(self,k_neightborh=5):
        self.tetanga=k_neightborh
    def fit(self,X,Y):
        self.X=X
        self.Y=Y
    def predict(self,X_test)->np.array:
        memo=[]
        for value in self.X:
            jarak=np.sum((self.X - value)**2,axis=1)
            rentang_indikasi=np.argsort(jarak)[:self.tetanga]
            rentang_y=self.Y[rentang_indikasi]
            memo.append(np.mean(rentang_y))
        return memo