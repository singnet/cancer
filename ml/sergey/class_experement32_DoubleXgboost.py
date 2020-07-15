import numpy as np
from xgboost import XGBClassifier


class DoubleXgboost:
    def __init__(self):
        self.clf01 = [XGBClassifier(), XGBClassifier()]
    
        
    def split(self,X,y):
        X01 = [[],[]]
        y01 = [[],[]]
        for Xi,yi in zip(X,y):
            idx = int(Xi[0])
            X01[idx].append(Xi[1:])
            y01[idx].append(yi)
        X01 = [np.array(X01[0]), np.array(X01[1])]
        y01 = [np.array(y01[0]), np.array(y01[1])]
        return X01, y01
        
    def fit(self, X,y):
        X01, y01 = self.split(X,y)
#        print(X01[0].shape, X01[1].shape, y01[0].shape, y01[1].shape)
        self.clf01[0].fit(X01[0], y01[0])
        self.clf01[1].fit(X01[1], y01[1])
    
        for i in range(2):
            assert self.clf01[i].classes_[1] == 1
        
        self.classes_ = self.clf01[0].classes_
        
        
    def predict(self, X):
        X01, idx01 = self.split(X, range(len(X)))
        
        y_pred = np.zeros(len(X))
        
        y_pred[idx01[0]] = self.clf01[0].predict(X01[0])
        y_pred[idx01[1]] = self.clf01[0].predict(X01[1])
        return y_pred
    
    def predict_proba(self, X):
        X01, idx01 = self.split(X, range(len(X)))
        
        y_pred_prob = np.zeros((len(X),2))
        
        for i in range(2):
            y_pred_prob[idx01[i]] = self.clf01[i].predict_proba(X01[i])
            
        return y_pred_prob
    
