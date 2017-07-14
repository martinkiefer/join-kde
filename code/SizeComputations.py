import Utils
import numpy as np

def ComputeAGMSSize(query,skn):
    icols = Utils.generateInvariantColumns(query)
    size_in_bit = 0
    
    #We have skn counter per table
    size_in_bit += skn * 32 * len(query.tables) 

    nicols = reduce(lambda x,y : x+len(y),icols,0)
    size_in_bit += skn*nicols*33
    
    size_in_bit += skn*(len(query.tables)-1)*33
    
    return (size_in_bit-1)/8+1
    #We need a 33-bit seed for every column


def ComputeAGMSSkn(query,size,local_size = 64):
    size = size * 8
    icols = Utils.generateInvariantColumns(query)
    nicols = reduce(lambda x,y : x+len(y),icols,0)
    #return size / (32 * len(query.tables)  + nicols*33 + (len(query.tables)-1)*33)
    
    size = (size / (32 * len(query.tables)  + nicols*33 + (len(query.tables)-1)*33))
    return size

def ComputeGPUKDESize(query,tuples):
    icols = Utils.generateInvariantColumns(query)
    nicols = reduce(lambda x,y : x+len(y),icols,0)
    
    size_in_bit = 0
    
    size_in_bit += 32*nicols*tuples
    size_in_bit += 32*len(query.joins)*tuples
    
    return size_in_bit/8
    
def ComputeGPUKDESampleSize(query,size):
    size *= 8
    icols = Utils.generateInvariantColumns(query)
    nicols = reduce(lambda x,y : x+len(y),icols,0)
    
    return size/(32*nicols+32*len(query.joins))
    
    
    
def ComputeGPUJKDESize(query,atuples):
    size_in_bit = 0
    
    for tid,t in enumerate(query.tables):
        size_in_bit += len(t.columns)*32*atuples[tid]
    return size_in_bit / 8
 
def ComputeGPUJKDESampleSize(query,sizes, local_size = 64):
    sizes = sizes*8
    factors = np.array(map(lambda x : len(x.columns)*32,query.tables))
    sizes = (sizes/factors)
    return sizes

def ComputeSumTableSize(query,stats):    
    size = 0
    for tid,t in enumerate(query.tables):
        size += ComputeSingleTableSize(query,tid,stats)
    return size
    
def ComputeSingleTableSize(query,tid,stats):
    ts, dv = stats
    return 4*len(query.tables[tid].columns)*ts[tid]
    
def ComputeAllTableSizes(query,stats):
    sizes = []
    for tid,t in enumerate(query.tables):
        sizes.append(ComputeSingleTableSize(query,tid,stats))
    return np.array(sizes)
    
