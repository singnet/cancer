import pandas as pd
from funs_common import read_alltreat_dataset
from collections import Counter

pd = read_alltreat_dataset()

c = Counter(zip(pd["study"], pd["platform_ID"]))

cs = Counter(pd["study"])


for study in sorted(set(pd["study"])):
    rez = []
    for platform_id in sorted(set(pd["platform_ID"])):
        if (c[(study, platform_id)] > 0):
            rez.append(platform_id)
    assert len(rez) == 1
    print(study, rez[0])
