import re
from collections import defaultdict
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='count_rez_auc_xgboost')
parser.add_argument('fn', help='filename')
args = parser.parse_args()

def read_f(fn):
    rez = defaultdict(dict)
    for l in open(fn):
        s = re.match("==> \((\w+)\) (study_\S+)", l)
        if (s):
            current_study = s[2]
            current_field = s[1]
            if (current_study not in rez[current_field]):
                rez[current_field][current_study] = {}
        s = re.match("==> \((\w+)\) (\S+)\s+:\s+(\S+) \(±(\S+)\)   (\S+) \(±(\S+)\)   (\S+) \(±(\S+)\)   (\S+) \(±(\S+)\)", l)        
        if (s):
            rez[current_field][current_study][s[2]] = [s[i] for i in range(3,3+8)]
    return rez

def count_rez_auc_xgboost(rez):
    for field in rez:
        total = 0
        more  = 0
        all_auc_fb = []
        all_auc_s  = []
        for study in rez[field]:
            auc_fb = float(rez[field][study]["withtrea_xgboost_fb"][6])
#            auc_fs = rez[field][study]["withtrea_xgboost_fs"][6]
            auc_s  = float(rez[field][study]["withtrea_xgboost_s"][6])
            
            all_auc_fb.append(auc_fb)
            all_auc_s.append(auc_s)
            total += 1
            more  += auc_fb > auc_s
            print(field, study, auc_fb, auc_s)
        print("total =", total,"  fb_better = ", more, "  mean_auc_fb=%.4g"%np.mean(all_auc_fb), "  mean_auc_s=%.4g"%np.mean(all_auc_s))
        print("")
        
rez = read_f(args.fn)

count_rez_auc_xgboost(rez) 
