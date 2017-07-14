import psycopg2 as pg
import numpy as np
import array
import os

def generateTemporaryHistogram(cur,table):
    #Create temp table histogram 
    first = True
    select = "create temp table t as select "
    for i,col in enumerate(table.columns):
        if not first:
            select += ", "
        select += "%s.%s as c%s" % (table.tid,col.cid,i+1)
        first = False
        
    select += ", count(*) as cnt from %s" % table.tid
    select += " group by "
    first = True
    for col in table.columns:
        if not first:
            select += ", "
        select += "%s.%s" % (table.tid,col.cid)
        first = False

    cur.execute(select)   
    
    #Create an index
    index_statement = "CREATE INDEX ON t ("
    first = True
    for i,col in enumerate(table.columns):
        if not first:
            index_statement += ", "
        index_statement += "c%s" % (i+1)
        first = False    
    index_statement += ");"
    cur.execute(index_statement)

def generateQueriesAndTrueSel(cur,table,queries):
    query_select = "select "
    first = True
    for i,c in enumerate(table.columns):
        if not first:
            query_select += ", "
        query_select += "%s.%s as c%s" % (table.tid,c.cid,i+1)
        first = False
           
    query_select += " from %s " % table.tid
    query_select += " order by random() limit %s" % queries
    
    full_gen_query = "select " 
    first = True
    for i,col in enumerate(table.columns):
        if not first:
            full_gen_query += ", "
        full_gen_query += "t.c%s" % (i+1)
        first = False

    full_gen_query += ", cnt from t, (%s) as q where " % query_select
    first = True
    for i,col in enumerate(table.columns):
        if not first:
            full_gen_query += " and "
        full_gen_query += "t.c%s = q.c%s" % (i+1,i+1)
        first = False

    full_gen_query += " order by "
    first = True
    for i,col in enumerate(table.columns):
        if not first:
            full_gen_query += ", "
        full_gen_query += "t.c%s" % (i+1)
        first = False
    
    cur.execute(full_gen_query)
    return cur.fetchall()
    
def generateQueryData(cnf,query,queries,fprefix,dprefix,iterations):        
    #Numpy arrays are super convenient to translate row-wise to columnar format
    for t in query.tables:
        con = pg.connect(cnf)
        cur = con.cursor()
        generateTemporaryHistogram(cur,t)
        for iteration in range(0,iterations):
            if not os.path.exists("%s/iteration%02d" % (fprefix,iteration)):
                os.makedirs("%s/iteration%02d" % (fprefix,iteration))
            np_array = np.array(generateQueriesAndTrueSel(cur,t,queries))
            for i,col in enumerate(t.columns):
                f = open("%s/iteration%02d/%stable_p_%s_%s.dump" % (fprefix,iteration,dprefix,t.tid,col.cid),"w")
                a = array.array("I",np_array.T[i])
                a.tofile(f)
                f.close()
            f = open("%s/iteration%02d/%stable_%s_true.dump" % (fprefix,iteration,dprefix,t.tid),"w")
            a = array.array("I",np_array.T[i+1])
            a.tofile(f)
            f.close()
        cur.close()
        con.close()