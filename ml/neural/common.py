import numpy
from collections import Counter, defaultdict


def mutual_information(p_x_y, p_x, p_y):
    mutual_info = 0
    x_values = set([item[0] for item in p_x_y.keys()])
    y_values = set([item[1] for item in p_x_y.keys()])
    for x in x_values:
        for y in y_values:
            if p_x_y[(x, y)]:
                mutual_info +=  p_x_y[(x, y)] * numpy.log2(p_x_y[(x, y)] / (p_x[x] * p_y[y]))
    return mutual_info


def compute_mi_stats(x, y):
    counter = Counter()
    counter.update(x)
    p_x = {k: v / len(x) for (k, v) in counter.items()}
    counter = Counter()
    counter.update(y)
    p_y = {k: v / len(y) for (k, v) in counter.items()}
    counter = Counter()
    assert(len(x) == len(y))
    counter.update(zip(x, y))
    p_x_y = defaultdict(int)
    p_x_y.update({k: v / len(x) for (k, v) in counter.items()})
    return p_x, p_y, p_x_y
