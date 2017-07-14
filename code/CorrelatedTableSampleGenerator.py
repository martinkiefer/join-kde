import psycopg2 as pg
import Utils
import numpy as np
import array
import os
import JoinGraph
    
def generateCorrelatedSample(cur,query,iteration,table,tid,size,jcols,sid,tsize):
    jmap = Utils.generateEquivalenceClassMap(query)

    query_select = "select "
    first = True
    for i,c in enumerate(table.columns):
        if not first:
            query_select += ", "
        query_select += "%s.%s as c%s" % (table.tid,c.cid,i+1)
        first = False
   
    #Compute target parameter.
    prob = (float(size)/tsize)**(1.0/len(jcols))

    #We need independent hash functions for every equivalence class.
    #We need to comnbine three values: iteration and equivalence class id

    query_select += " from %s where " % table.tid
    first = True
    for jcol in jcols:
        seed = iteration*1000+jmap[(tid,jcol)]
        hc = Utils.Uhash(seed)
        if not first:
            query_select += " and "
        query_select += "CAST( ((CAST(%s.%s as BIGINT) * %s + %s) %% %s) as float4) / (%s-1) < %f " % (table.tid,table.columns[jcol].cid,hc.a,hc.b,hc.p,hc.p,prob)
        first = False
        
    query_select += " order by c%s" % (sid+1)
    cur.execute(query_select)
    return cur.fetchall()
    
def generateCorrelatedSamples(cnf,query,sample_sizes,fprefix,dprefix,iterations,stats):        
    #Numpy arrays are super convenient to translate row-wise to columnar format
    ts, dv = stats
    jcols = Utils.generateJoinColumns(query)
    graph = JoinGraph.constructJoinGraph(query)
    sidd = {}
    graph.getSortColumnDict(sidd)

    for par in sample_sizes:
        for tid,size in enumerate(par):
            t = query.tables[tid]
            con = pg.connect(cnf)
            cur = con.cursor()
            for iteration in range(0,iterations):
                if not os.path.exists("%s/iteration%02d" % (fprefix,iteration)):
                    os.makedirs("%s/iteration%02d" % (fprefix,iteration))
                    
                np_array = np.array(generateCorrelatedSample(cur,query,iteration,t,tid,size,jcols[tid],sidd[tid],ts[tid]))
                for i,col in enumerate(t.columns):
                    f = open("%s/iteration%02d/%scsample_%s_%s_%s.dump" % (fprefix,iteration,dprefix,size,t.tid,col.cid),"w")
                    if len(np_array) > 0:
                        a = array.array("I",np_array.T[i])
                        a.tofile(f)
                    f.close()
        cur.close()
        con.close()
