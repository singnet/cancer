import numpy as np
from xgboost import XGBClassifier
from collections import Counter,defaultdict

class BiasedXgboost:
    def __init__(self):
        self.clf = XGBClassifier()
    
    def fit(self, X,y):
        self.clf.fit(X[:,1:], y)
        c = Counter(zip(X[:,0],y))
        self.Pset0 = np.zeros(2)
        self.Pset1 = np.zeros(2)
        
        self.Pset0[0] = c[(0,0)] / (c[0,0] + c[1,0])
        self.Pset0[1] = c[(1,0)] / (c[0,0] + c[1,0])
        
        
        self.Pset1[0] = c[(0,1)] / (c[0,1] + c[1,1])
        self.Pset1[1] = c[(1,1)] / (c[0,1] + c[1,1])
#        print("Pset", self.Pset)
        assert self.clf.classes_[1] == 1
        self.classes_ = self.clf.classes_
        
        
    def predict(self, X):
        return np.array(self.predict_proba(X)[:,1] > 0.5, dtype=np.int)
    
    def predict_proba(self, X):
        proba = self.clf.predict_proba(X[:,1:])
        
        rez = []
        for probai, s in zip(proba, X[:,0]):
            # P(D|F2) = probai[1]
            # P(F1|D) = self.Pset[int(s)]
            # P(!D|F2) = 1 - probai[1]
            # P(F1|!D) = 1 - self.Pset[int[s]]
            p1 = probai[1] * self.Pset1[int(s)] / (probai[1] * self.Pset1[int(s)] + probai[0] * self.Pset0[int(s)])
            p0 = probai[0] * self.Pset0[int(s)] / (probai[0] * self.Pset0[int(s)] + probai[1] * self.Pset1[int(s)])
            
#            print(p1, p0, p1 + p0)
#            exit(0)
            rez.append((p0,p1))
#            print((probai[1] * self.Pset[int(s)] + (1 - probai[1]) * (1 - self.Pset[int(s)])))
#            print(p0, p1, p0 + p1)
        
        return np.array(rez)
    
