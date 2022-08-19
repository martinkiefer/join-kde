import psycopg2 as pg
import numpy as np
import array
import os
from Utils import generateInvariantColumns

def generateJoinSample(cur,query,cols,size):
    first = True
    query_select = "select "
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                query_select += ", "
            query_select += "%s.%s as c%s" % (query.tables[i].tid,query.tables[i].columns[c].cid,j)
            j += 1
            first = False
    first = True           
    query_select += " from "
    for t in query.tables:
        if not first:
            query_select += ","
        query_select += " %s" % t.tid
        first = False
        
    query_select += " where "
    first = True
    for j,join in enumerate(query.joins):
        bt, bc = join[0]
        for i,tabcol in enumerate(join[1:]):
            if not first:
                query_select += " and "
            a,b = tabcol
            query_select += "%s.%s = %s.%s" % (query.tables[a].tid,query.tables[a].columns[b].cid,query.tables[bt].tid,query.tables[bt].columns[bc].cid)
            first = False

    query_select += " order by random() limit %s" % size
    
    cur.execute(query_select)
    return cur.fetchall()
    
def generateSamples(cnf,query,sample_sizes,fprefix,dprefix,iterations):    
    cols = generateInvariantColumns(query)
    con = pg.connect(cnf)
    cur = con.cursor()

    #Numpy arrays are super convenient to translate row-wise to columnar format
    for iteration in range(0,iterations):
        if not os.path.exists("%s/iteration%02d" % (fprefix,iteration)):
            os.makedirs("%s/iteration%02d" % (fprefix,iteration))
        for size in sample_sizes:
            np_array = np.array(generateJoinSample(cur,query,cols,size))
            j = 0
            for i,col in enumerate(cols):
                for c in col:
                    f = open("%s/iteration%02d/%sjsample_%s_%s_%s.dump" % (fprefix,iteration,dprefix,size,query.tables[i].tid,query.tables[i].columns[c].cid),"wb")
                    a = array.array("I",np_array.T[j])
                    a.tofile(f)
                    f.close()
                    j += 1  
    cur.close()
    con.close()
