import numpy as np
from collections import defaultdict
import random

def random_upsample_balance(X,y):
    assert len(X) == len(y)
    Xsplit = defaultdict(list)
    for Xi,yi in zip(X,y):
        Xsplit[yi].append(Xi)
    
    max_len = max(len(Xyi) for _, Xyi in Xsplit.items())
    
    for yi,Xyi in list(Xsplit.items()):
        add_Xyi = random.choices(Xyi, k = max_len - len(Xyi))
        Xsplit[yi] = Xyi + add_Xyi 
    
    X_rez = []
    y_rez = []
    for yi, Xyi in Xsplit.items():
        X_rez = X_rez + Xyi
        y_rez = y_rez + [yi] * len(Xyi)
    return np.array(X_rez),np.array(y_rez)

