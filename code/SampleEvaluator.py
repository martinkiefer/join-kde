# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:48:31 2016

@author: mkiefer
"""
import Utils 
import sys
from collections import defaultdict
import copy
import pickle
from functools import reduce

#A sort of columnar join
class JoinNode:
    def __init__(self,query,iteration,sample_sizes,left,left_col,right,right_col):
        self.left_col = left_col
        self.right_col = right_col
        self.left = left
        self.right = right
        self.ssl = sample_sizes[self.left_col[0]]
        self.ssr = sample_sizes[self.right_col[0]]
        self.iteration = iteration
        self.query = query

    def next_query(self):
        self.left.next_query()
        self.right.next_query()    
        
    def get(self):
        rtids = self.right.get()  
        rfile = "iteration%02d/sample_%s_%s_%s.dump" % (self.iteration,self.ssr,self.query.tables[self.right_col[0]].tid,self.query.tables[self.right_col[0]].columns[self.right_col[1]].cid)
        rcol = Utils.readCol(rfile,self.ssr)
        
        rvals = gather(rcol,rtids[self.right_col[0]])
        
        #Build a hash table
        ht = defaultdict(list)
        for i,v in enumerate(rvals):
            ht[v].append(i)
            
        ltids = self.left.get()  
        lfile = "iteration%02d/sample_%s_%s_%s.dump" % (self.iteration,self.ssl,self.query.tables[self.left_col[0]].tid,self.query.tables[self.left_col[0]].columns[self.left_col[1]].cid)
        lcol = Utils.readCol(lfile,self.ssl)   
        
        lvals = gather(lcol,ltids[self.left_col[0]])
        
        #And now probe that fucker
        res = defaultdict(list)
        for i,v in enumerate(lvals):
            for vh in ht[v]:
                for k,v in rtids.items():
                    res[k].append(v[vh])
                for k,v in ltids.items():
                    res[k].append(v[i])       
        return res
    
    def __str__(self):
        return "join(%s = %s, %s, %s)" % (self.left_col,self.right_col,self.left,self.right) 
        
class SampleScan:
    def __init__(self,query,iteration,sample_sizes,test,table):
        self.table = table
        self.query = query
        self.ss = sample_sizes[self.table]
        self.iteration = iteration
        self.ctr = 0
        self.test = test
        
        self.selections = {}
        cols = Utils.generateInvariantColumns(self.query)
        for c in cols[self.table]:
            if self.query.tables[self.table].columns[c].type == "point":
                filename = "iteration%02d/test_join_p_%s_%s.dump" % (self.iteration,self.query.tables[self.table].tid,self.query.tables[self.table].columns[c].cid)
                self.selections[(c,)] = Utils.readCol(filename,self.test)
            elif self.query.tables[self.table].columns[c].type == "range":
                filename = "iteration%02d/test_join_l_%s_%s.dump" % (self.iteration,self.query.tables[self.table].tid,self.query.tables[self.table].columns[c].cid)
                self.selections[(c,'l')] = Utils.readCol(filename,self.test)
                filename = "iteration%02d/test_join_u_%s_%s.dump" % (self.iteration,self.query.tables[self.table].tid,self.query.tables[self.table].columns[c].cid)
                self.selections[(c,'u')] = Utils.readCol(filename,self.test)
                
    def get(self):
        icols = Utils.generateInvariantColumns(self.query)
        
        tids = list(range(0,self.ss))
        for c in icols[self.table]:
            if self.query.tables[self.table].columns[c].type == "point":
                tids = getMatchingOffsetsInclusiveRange(self.query, self.table,c,self.iteration,self.ss,self.selections[(c,)][self.ctr],self.selections[(c,)][self.ctr],tids)
            elif self.query.tables[self.table].columns[c].type == "range":
                tids = getMatchingOffsetsInclusiveRange(self.query,self.table,c,self.iteration,self.ss,self.selections[(c,'l')][self.ctr],self.selections[(c,'u')][self.ctr],tids)
        return {self.table : tids}
        
    def next_query(self):
        self.ctr += 1
    
    def __str__(self):
        return "SS(%s)" % str(self.table)

def getMatchingOffsetsInclusiveRange(query, table, column, iteration, ss, l,u, tids=None):
    filename = "iteration%02d/sample_%s_%s_%s.dump" % (iteration,ss,query.tables[table].tid,query.tables[table].columns[column].cid)
    col = Utils.readCol(filename,ss)
    if tids == None:
        return [i for i in range(0,ss) if col[i] <= u and col[i] >= l]
    else:
        return [i for i in tids if col[i] <= u and col[i] >= l]
        
    
def gather(vals,indices):
    return [vals[x] for x in indices]                
                
def constructJoinGraph(query,iteration,sample_sizes,test):
    tables = [set() for _ in query.joins]
    
    joins = copy.deepcopy(query.joins)
    
    for i,join in enumerate(query.joins):   
        for j in join:
            tables[i].add(j[0])
    #Start off with the initial join
    jc = tuple(joins[0][0])
    tree = SampleScan(query,iteration,sample_sizes,test,jc[0])
    
    for j in joins[0][1:]:
        tree = JoinNode(query,iteration,sample_sizes,tree,jc,SampleScan(query,iteration,sample_sizes,test,j[0]),tuple(j))
    tree_tables = tables.pop(0)
    joins.pop(0)
    
    while joins:
        for i,(join,table) in enumerate(zip(joins,tables)):
            #Okay, this join works together with our previous join tree.
            if tree_tables & table:
                #Loop over all subjoins
                first = True
                #The column required from the other join result
                col_left = None
                jc = None
                for tcol in join:
                    x,y = tcol
                    #We found the column from the other join.
                    if x in tree_tables:
                        col_left = (x,y)
                        continue
                    elif first:
                        rtree = SampleScan(query,iteration,sample_sizes,test,x)
                        jc = (x,y)
                        first = False
                    else:
                        rtree = JoinNode(query,iteration,sample_sizes,rtree,jc,SampleScan(query,iteration,sample_sizes,test,x),(x,y))  
                
                tree = JoinNode(query,iteration,sample_sizes,tree,col_left,rtree,jc)
                tree_tables |= tables.pop(i)
                joins.pop(i)
                break
    return tree
    
#Quick Bernoulli Estimator

if __name__ == '__main__':  
	with open('./descriptor.json', 'r') as f:
		descriptor = Utils.json2obj(f.read())    
		query = descriptor.query_descriptor

		
		tabs = len(query.tables)
		sample_sizes = [0]*tabs
		for i in range(1,tabs+1):
			sample_sizes[i-1] = int(sys.argv[i])
		iteration=int(sys.argv[tabs+1])
		test = int(sys.argv[tabs+2])
		
		with open("./stats.pick",'r') as f:
			ts,_ = pickle.load(f)
		
		true_cardinalities = Utils.readCol("iteration%02d/test_join_true.dump" % iteration,test)
		scale = reduce(lambda x,y : x*float(y[0])/float(y[1]),list(zip(list(ts.values()),sample_sizes)),1.0)
		
		g = constructJoinGraph(query,iteration,sample_sizes,test)    
		
		for i in range(0,test):
			Utils.printResultString(sum(sample_sizes),scale*len(g.get()[0]),true_cardinalities[i],iteration)
			g.next_query()
		
