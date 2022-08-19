# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:48:31 2016

@author: mkiefer
"""
import psycopg2 as pg
import Utils 
import sys
from collections import defaultdict
import pickle
import re

def constructJoinQuery(query):
     icols = Utils.generateInvariantColumns(query)
     s = "explain select * from %s where" % ",".join([x.tid for x in query.tables])

     jps = []
     for j in query.joins:
         first=None
         for t,c in j:
              tup = (query.tables[t].tid,query.tables[t].columns[c].cid)
              if first is None:
                  first = tup           
              else:
                   jps.append((first,tup)) 

     s += "and".join([" %s.%s = %s.%s " % (x[0][0],x[0][1],x[1][0],x[1][1]) for x in jps])    
     #for t,cs in enumerate(icols):
     #   for c in cs:
     # 	    s += " and %s.%s = %%s" % (query.tables[t].tid,query.tables[t].columns[c].cid)
     return s


with open('./descriptor.json', 'r') as f:
    descriptor = Utils.json2obj(f.read())    
    query = descriptor.query_descriptor
    
    icols = Utils.generateInvariantColumns(query)
    jcols = Utils.generateJoinColumns(query)
   
    iteration=int(sys.argv[1])
    test = int(sys.argv[2])

    con = pg.connect(descriptor.pg_conf) 

    cur = con.cursor()
    for t in query.tables:
        cur.execute("ANALYZE %s;" % t.tid)
    
    query_string = constructJoinQuery(query)
    true_cardinalities = Utils.readCol("iteration%02d/test_join_true.dump" % iteration,test)
    predicates = {}
    for t,cs in enumerate(icols):
        for c in cs:
            if query.tables[t].columns[c].type == "range":
                predicates[(t,c,'u')]=Utils.readCol("iteration%02d/test_join_u_%s_%s.dump" % (iteration,query.tables[t].tid,query.tables[t].columns[c].cid),test)
                predicates[(t,c,'l')]=Utils.readCol("iteration%02d/test_join_l_%s_%s.dump" % (iteration,query.tables[t].tid,query.tables[t].columns[c].cid),test)
            elif query.tables[t].columns[c].type == "point":
                predicates[(t,c,'p')]=Utils.readCol("iteration%02d/test_join_p_%s_%s.dump" % (iteration,query.tables[t].tid,query.tables[t].columns[c].cid),test)
            else:
                raise Exception("Woot woot?")
    
    ex = re.compile(".*(Join|Loop).*rows=(\d+)")
    for i in range(0,test):
        qs = query_string
        for t,cs in enumerate(icols):
            for c in cs:
                if query.tables[t].columns[c].type == "range":
                    qs += " and %s.%s >= %s" % (query.tables[t].tid,query.tables[t].columns[c].cid,predicates[(t,c,'l')][i])
                    qs += " and %s.%s <= %s" % (query.tables[t].tid,query.tables[t].columns[c].cid,predicates[(t,c,'u')][i])
                elif query.tables[t].columns[c].type == "point":
                    qs += " and %s.%s = %s" % (query.tables[t].tid,query.tables[t].columns[c].cid,predicates[(t,c,'p')][i])
                else:
                    raise Exception("Woot woot?")
        cur.execute(qs)
        #Now find the join
        m = None
        for row in cur.fetchall():
            m = ex.match(row[0])
            if m != None:
                break
        if m == None:
            raise Exception("There was no join in query %s" % qs)
        Utils.printResultString(88,float(m.group(2)),true_cardinalities[i],iteration)
    cur.close()
    con.close()
