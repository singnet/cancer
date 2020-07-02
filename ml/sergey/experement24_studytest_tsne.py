import pandas as pd
import os
import numpy as np
import random
import math 
from funs_common import read_mike_dataset, drop_trea, prepare_full_dataset
from sklearn.manifold import TSNE
import pickle 

dataset        = read_mike_dataset()

notrea_dataset = drop_trea(dataset)
X,y     = prepare_full_dataset(notrea_dataset, y_field = 'study')

X_embedded = TSNE(n_components=2).fit_transform(X)

pickle.dump(X_embedded, open( "experement24_studytest_tsne.p", "wb" ) )
