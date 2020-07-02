from funs_common import read_treat_dataset
from collections import Counter

def count_val(ds, val):
    c = Counter(ds[val])
    return c[0], c[1], len(ds[val]) - c[0] - c[1]

def count_pcr_rfs_dfs(ds):
    print("pCR", *count_val(ds, "pCR"))
    print("RFS", *count_val(ds, "RFS"))
    print("DFS", *count_val(ds, "DFS"))
    
d = read_treat_dataset()

all_studies = list(set(d['study']))

for study in all_studies:
    print(study)
    ds = d.loc[d['study'] == study]
    count_pcr_rfs_dfs(ds)
    print("")
