import itertools
import sys
import numpy as np

with open(sys.argv[1]) as f:
    lines = map(lambda x : np.array(map(lambda y : float(y),x.split(','))), f.readlines())
    data = sorted(lines, key=lambda x : (x[1]))
         
    i = 0
    for k,g in itertools.groupby(data, key=lambda x : (x[1])):
         vals = []
         l = np.array(map(lambda x : x[1:],g))
         
         val = np.mean(l,axis=0)
         print "%s,%s" % (0.001*2**i,','.join(map(lambda x: str(x), val)))
         i += 1
