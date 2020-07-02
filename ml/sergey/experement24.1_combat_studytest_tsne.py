import pandas as pd
import os
import numpy as np
import random
import math 
from funs_common import read_combat_dataset, drop_trea, prepare_full_dataset
from sklearn.manifold import TSNE
import pickle 
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def get_pam_names(notrea_dataset):
    coincide_types = pd.read_csv('coincideTypes.csv')

    ids_before = notrea_dataset["patient_ID"]
    pam_types = pd.merge(notrea_dataset["patient_ID"],coincide_types[["patient_ID", "pam_name"]])
    ids_after = pam_types["patient_ID"]
    assert all(ids_before == ids_after)
    return pam_types["pam_name"].to_numpy()
    
def read_coincide_types_dataset(path="./"):
    dataset = pd.read_csv(os.path.join(path, 'coincideTypes.csv'))
    treat_dataset = read_treat_dataset()
    return pd.merge(treat_dataset, dataset)
            

def y_to_int(y):
    ydict = {s:i for i,s in (enumerate(set(y)))}
    for k,v in ydict.items():
        print(k, v)
    return np.array([ydict[yi] for yi in y])

def remove_nans(l):
    return [li for li in l if str(li) != 'nan']

def plot_vs_y(y, X_embedded, figname):
    all_studies = list(set(y))
    all_studies = remove_nans(all_studies)
    print(all_studies)
    plt.clf() 
    for study in all_studies:
        X_emb_x = X_embedded[y == study][:,0]
        X_emb_y = X_embedded[y == study][:,1]
        plt.scatter(X_emb_x, X_emb_y, s = 3, alpha = 0.5, label = study)
    
    fontP = FontProperties()
    fontP.set_size('x-small')
    plt.legend(prop=fontP)

    plt.savefig(figname, dpi=400)
    

dataset = read_combat_dataset()
notrea_dataset = drop_trea(dataset)
X,y     = prepare_full_dataset(notrea_dataset, y_field = 'study')

y = y_to_int(y)

X_embedded = TSNE(n_components=2).fit_transform(X)


plot_vs_y(y, X_embedded, "experement24.1_combat_study.png")

pam_names = get_pam_names(notrea_dataset)
plot_vs_y(pam_names, X_embedded, "experement24.1_combat_pam_names.png")

plot_vs_y(dataset["pCR"].to_numpy(), X_embedded, "experement24.1_combat_pCR.png")
plot_vs_y(dataset["DFS"].to_numpy(), X_embedded, "experement24.1_combat_DFS.png")
plot_vs_y(dataset["RFS"].to_numpy(), X_embedded, "experement24.1_combat_RFS.png")
plot_vs_y(dataset["posOutcome"].to_numpy(), X_embedded, "experement24.1_combat_posOutcome.png")
plot_vs_y(dataset["radio"].to_numpy(), X_embedded, "experement24.1_combat_radio.png")
plot_vs_y(dataset["surgery"].to_numpy(), X_embedded, "experement24.1_combat_surgey.png")
plot_vs_y(dataset["chemo"].to_numpy(), X_embedded, "experement24.1_combat_chemo.png")
plot_vs_y(dataset["hormone"].to_numpy(), X_embedded, "experement24.1_combat_hormone.png")

