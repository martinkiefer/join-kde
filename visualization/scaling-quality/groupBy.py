import itertools
import sys
import numpy as np
from scipy.stats.mstats import gmean

with open(sys.argv[1]) as f:
    lines = [np.array([float(y) for y in x.split(',')]) for x in f.readlines()]
    data = sorted(lines, key=lambda x : (x[1]))
         
    i = 0
    for k,g in itertools.groupby(data, key=lambda x : (x[1])):
        vals = []
        l = np.array([x[1:] for x in g])
        print(l)

        val = gmean(l,axis=0)#**(1/l.shape[1])
        print("%s,%s" % (0.001*2**i,','.join([str(x) for x in val])))
        i += 1
