# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:48:31 2016

@author: mkiefer
"""
import Utils 
import sys
from collections import defaultdict
import cPickle

        
class JoinSampleScan:
    def __init__(self,query,iteration,sample_size,test):
        self.query = query
        self.ss = sample_size
        self.iteration = iteration
        self.ctr = 0
        self.test = test
        
        self.selections = {}
        icols = Utils.generateInvariantColumns(self.query)
        
        for table,cols in enumerate(icols):
            for c in cols:
                if self.query.tables[table].columns[c].type == "point":
                    filename = "iteration%02d/test_join_p_%s_%s.dump" % (self.iteration,self.query.tables[table].tid,self.query.tables[table].columns[c].cid)
                    self.selections[(table,c)] = Utils.readCol(filename,self.test)
                    
                elif self.query.tables[table].columns[c].type == "range":
                    filename = "iteration%02d/test_join_l_%s_%s.dump" % (self.iteration,self.query.tables[table].tid,self.query.tables[table].columns[c].cid)
                    self.selections[(table,c,'l')] = Utils.readCol(filename,self.test)
                    filename = "iteration%02d/test_join_u_%s_%s.dump" % (self.iteration,self.query.tables[table].tid,self.query.tables[table].columns[c].cid)
                    self.selections[(table,c,'u')] = Utils.readCol(filename,self.test)
                
    def get(self):
        icols = Utils.generateInvariantColumns(self.query)
        
        tids = list(range(0,self.ss))
        for table,cols in enumerate(icols):
            for c in cols:
                if self.query.tables[table].columns[c].type == "point":
                    tids = getMatchingOffsetsInclusiveRange(self.query, table,c,self.iteration,self.ss,self.selections[(table,c)][self.ctr],self.selections[(table,c)][self.ctr],tids)
                elif self.query.tables[table].columns[c].type == "range":
                    tids = getMatchingOffsetsInclusiveRange(self.query,table,c,self.iteration,self.ss,self.selections[(table,c,'l')][self.ctr],self.selections[(table,c,'u')][self.ctr],tids)
        return tids
        
    def next_query(self):
        self.ctr += 1
    
    def __str__(self):
        return "SS(%s)" % str(self.table)

def getMatchingOffsetsInclusiveRange(query, table, column, iteration, ss, l,u, tids=None):
    filename = "iteration%02d/jsample_%s_%s_%s.dump" % (iteration,ss,query.tables[table].tid,query.tables[table].columns[column].cid)
    col = Utils.readCol(filename,ss)
    if tids == None:
        return [i for i in range(0,ss) if col[i] <= u and col[i] >= l]
    else:
        return [i for i in tids if col[i] <= u and col[i] >= l]
        
    
def gather(vals,indices):
    return map(lambda x : vals[x],indices)                
                
    
#Quick Bernoulli Estimator
  
with open('./descriptor.json', 'r') as f:
    descriptor = Utils.json2obj(f.read())    
    query = descriptor.query_descriptor
    
    icols = Utils.generateInvariantColumns(query)
    jcols = Utils.generateJoinColumns(query)
    
    tabs = len(query.tables)
    sample_size = int(sys.argv[1])
    iteration=int(sys.argv[2])
    test = int(sys.argv[3])
    
    with open("./jstats.pick",'r') as f:
        ts,_ = cPickle.load(f)
    
    true_cardinalities = Utils.readCol("iteration%02d/test_join_true.dump" % iteration,test)
    scale = float(ts)/sample_size
    
    jss = JoinSampleScan(query,iteration,sample_size,test)    
    
    for i in range(0,test):
        Utils.printResultString(sample_size,scale*len(jss.get()),true_cardinalities[i],iteration)
        jss.next_query()
    
