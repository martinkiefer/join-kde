import psycopg2 as pg
import numpy as np
import array
import os
from Utils import generateInvariantColumns
    
def generateTemporaryHistogram(cur,query,cols):
    #Create temp table histogram 
    first = True
    select = "create temp table t as select "
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                select += ", "
            select += "%s.%s as c%s" % (query.tables[i].tid,query.tables[i].columns[c].cid,j)
            j += 1
            first = False
    first = True           
    select += ", count(*) as cnt from "
    for t in query.tables:
        if not first:
            select += ","
        select += " %s" % t.tid
        first = False
        
    select += " where "
    first = True
    for j,join in enumerate(query.joins):
        bt, bc = join[0]
        for i,tabcol in enumerate(join[1:]):
            if not first:
                select += " and "
            a,b = tabcol
            select += "%s.%s = %s.%s" % (query.tables[a].tid,query.tables[a].columns[b].cid,query.tables[bt].tid,query.tables[bt].columns[bc].cid)
            first = False

    select += " group by "
    first = True
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                select += ", "
            select += "%s.%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
            first = False
    cur.execute(select)   
    
    #Create an index
    index_statement = "CREATE INDEX ON t ("
    j = 1
    first = True
    for col in cols:
        for c in col:
            if not first:
                index_statement += ", "
            index_statement += ("c%s" % (j))
            j += 1
            first = False    
    index_statement += ");"
    cur.execute(index_statement)

def generateQueriesAndTrueSel(cur,query,cols,queries):
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

    query_select += " order by random() limit %s" % queries
    
    full_gen_query = "select " 
    first = True
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                full_gen_query += ", "
            full_gen_query += "t.c%s" % (j)
            j += 1
            first = False

    full_gen_query += ", cnt from t, (%s) as q where " % query_select
    first = True
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                full_gen_query += " and "
            full_gen_query += "t.c%s = q.c%s" % (j,j)
            j += 1
            first = False

    full_gen_query += " order by "
    first = True
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                full_gen_query += ", "
            full_gen_query += "t.c%s" % (j)
            j += 1
            first = False

    cur.execute(full_gen_query)
    return cur.fetchall()
    
def generateQueryData(cnf,query,queries,fprefix,dprefix,iterations):    
    con = pg.connect(cnf)
    cur = con.cursor()

    cols = generateInvariantColumns(query)
    generateTemporaryHistogram(cur,query,cols)
    
    #Numpy arrays are super convenient to translate row-wise to columnar format
    for iteration in range(0,iterations):
        if not os.path.exists("%s/iteration%02d" % (fprefix,iteration)):
            os.makedirs("%s/iteration%02d" % (fprefix,iteration))
        np_array = np.array(generateQueriesAndTrueSel(cur,query,cols,queries))
        j = 0
        for i,col in enumerate(cols):
            for c in col:
                f = open("%s/iteration%02d/%sjoin_p_%s_%s.dump" % (fprefix,iteration,dprefix,query.tables[i].tid,query.tables[i].columns[c].cid),"w")
                a = array.array("I",np_array.T[j])
                a.tofile(f)
                f.close()
                j += 1  
        f = open("%s/iteration%02d/%sjoin_true.dump" % (fprefix,iteration,dprefix),"w")
        a = array.array("I",np_array.T[j])
        a.tofile(f)
        f.close()
    cur.close()
    con.close()