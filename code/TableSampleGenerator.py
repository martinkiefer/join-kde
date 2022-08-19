import psycopg2 as pg
import numpy as np
import array
import os
import JoinGraph
    
def generateSample(cur,table,size,sid):
    query_select = "select * from (select "
    first = True
    for i,c in enumerate(table.columns):
        if not first:
            query_select += ", "
        query_select += "%s.%s as c%s" % (table.tid,c.cid,i+1)
        first = False
           
    query_select += " from %s " % table.tid
    query_select += " order by random() limit %s) as meh order by c%s" % (size,sid+1)
    cur.execute(query_select)
    return cur.fetchall()
    
def generateSamples(cnf,query,sample_sizes,fprefix,dprefix,iterations):        
    #Numpy arrays are super convenient to translate row-wise to columnar format
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
                    
                np_array = np.array(generateSample(cur,t,size,sidd[tid]))
                for i,col in enumerate(t.columns):
                    f = open("%s/iteration%02d/%ssample_%s_%s_%s.dump" % (fprefix,iteration,dprefix,size,t.tid,col.cid),"wb")
                    a = array.array("I",np_array.T[i])
                    a.tofile(f)
                    f.close()
        cur.close()
        con.close()
