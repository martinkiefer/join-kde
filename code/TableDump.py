import psycopg2 as pg
import numpy as np
import array
    
def dumpTable(cur,table):
    query_select = "select "
    first = True
    for c in table.columns:
        if not first:
            query_select += ", "
        query_select += "%s.%s" % (table.tid,c.cid)
        first = False
           
    query_select += " from %s " % table.tid
    
    cur.execute(query_select)
    return cur.fetchall()
    
def dumpTables(cnf,query,fprefix,dprefix):        
    #Numpy arrays are super convenient to translate row-wise to columnar format
    con = pg.connect(cnf)
    cur = con.cursor()
    for t in query.tables:
        np_array = np.array(dumpTable(cur,t))
        for i,col in enumerate(t.columns):
            f = open("%s/%stable_%s_%s.dump" % (fprefix,dprefix,t.tid,col.cid),"w")
            a = array.array("I",np_array.T[i])
            a.tofile(f)
            f.close()
    cur.close()
    con.close()