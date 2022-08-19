import psycopg2 as pg
import numpy as np
import array
import os
from Utils import generateInvariantColumns
from Utils import generateInvariantPointColumns
from Utils import generateInvariantRangeColumns
from Utils import generateJoinColumns
from Utils import flatten


def generateMinMaxQueryString(cur,query):
    #Create temp table histogram 

    cols = generateInvariantRangeColumns(query)

    first = True
    select = "select "
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                select += ", "
            select += "min(t%s_c%s) as minc%s, " % (query.tables[i].tid,query.tables[i].columns[c].cid,j)
            select += "max(t%s_c%s) as maxc%s" % (query.tables[i].tid,query.tables[i].columns[c].cid,j)
            j += 1
            first = False
    first = True           
    select += " from t "
        
    return select


def generateJoinQueryString(cur,query):
    #Create temp table histogram 

    cols = generateInvariantRangeColumns(query)

    first = True
    select = "select sum(cnt) "
    select += " from t "
        
    select += " where "
    return select

def generateTemporaryHistogram(cur,query,pcols,rcols,workload):
    #Create temp table histogram 
    
    #Unique version
    if workload == "distinct":
        select = "create table t as select row_number() over () as rnum,"
    elif workload == "uniform":
        select = "create table t as select sum(count(*)) over (order by "
        first = True
        j = 1
        for i,col in enumerate(pcols):
            for c in col:
                if not first:
                    select += ", "
                select += "%s.%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
                j += 1
                first = False
        for i,col in enumerate(rcols):
            for c in col:
                if not first:
                    select += ", "
                select += "%s.%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
                j += 1
                first = False
        select += ") as rnum, "
    else:
        raise Exception("Unknown workload type.")



    first = True
    j = 1
    for i,col in enumerate(pcols):
        for c in col:
            if not first:
                select += ", "
            select += "%s.%s as t%s_c%s" % (query.tables[i].tid,query.tables[i].columns[c].cid,query.tables[i].tid,query.tables[i].columns[c].cid)
            j += 1
            first = False
    for i,col in enumerate(rcols):
        for c in col:
            if not first:
                select += ", "
            select += "%s.%s as t%s_c%s" % (query.tables[i].tid,query.tables[i].columns[c].cid,query.tables[i].tid,query.tables[i].columns[c].cid)
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
    for i,col in enumerate(pcols):
        for c in col:
            if not first:
                select += ", "
            select += "%s.%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
            first = False
    for i,col in enumerate(rcols):
        for c in col:
            if not first:
                select += ", "
            select += "%s.%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
            first = False
    cur.execute(select)   
    
    #Create an index
    index_statement = "CREATE INDEX ON t(rnum);"
    cur.execute(index_statement)

    #Create an index
    index_statement = "CREATE INDEX ON t ("
    first = True
    j = 1
    for i,col in enumerate(pcols):
        for c in col:
            if not first:
                index_statement += ", "
            index_statement += "t%s_c%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
            j += 1
            first = False
    for i,col in enumerate(rcols):
        for c in col:
            if not first:
                index_statement += ", "
            index_statement += "t%s_c%s" % (query.tables[i].tid,query.tables[i].columns[c].cid)
            j += 1
            first = False

    index_statement += ");"
    cur.execute(index_statement)

    if workload == "uniform":
        cur.execute("select sum(cnt) from t;")
    elif workload == "distinct":
        cur.execute("select count(*) from t;")

    rnums = cur.fetchone()[0]
    return int(rnums)

def generateQueriesAndTrueSel(cur,query,cols,queries,workload="uniform",maxrnum=1):
    first = True

    pcols = generateInvariantPointColumns(query)
    pcolsf = [ (query.tables[i].tid,query.tables[i].columns[j].cid) for i in range(len(pcols)) for j in pcols[i] ]
    
    pattern = " "
    first = True
    if len(pcolsf) > 0:
        for tid,cid in pcolsf:
            if first:
                pattern +=" t%s_c%s = %%s " % (tid,cid)
                first = False
            else:
                pattern +=" and t%s_c%s = %%s " % (tid,cid)
        #cur.execute("select * from t;")
        #histogram = np.array(cur.fetchall())
        #Place histogram lock here.
        #if workload == "distinct":
        #    freqs = histogram[:,-1]/float(np.sum(histogram[:,-1]))
        #elif workload == "uniform":
        #    freqs = np.ones(histogram[:,-1].shape)/len(histogram[:,-1])
        #else:
        #    raise Exception("This workload does not exist.")

    rcols = generateInvariantRangeColumns(query)
    rcolsf = [ (query.tables[i].tid,query.tables[i].columns[j].cid) for i in range(len(rcols)) for j in rcols[i] ]
    nrcols = len(rcolsf)

    minpattern = " "
    maxpattern = " "
    if len(rcolsf) > 0:
        for tid,cid in rcolsf:
            if first:
                minpattern +=" t%s_c%s >= %%s " % (tid,cid)
                first = False
            else:
                minpattern +=" and t%s_c%s >= %%s " % (tid,cid)
            maxpattern +=" and t%s_c%s <= %%s " % (tid,cid)


    rejected = 0
    pvals = []
    uvals = []
    lvals = []
    cards = []
    minmaxq_prev = None
    minv = None
    maxv = None
    card = None
    while len(cards) < queries:
        #We start by drawing a random cluster center. Right now, we let it 
        if True:
            #vals = histogram[np.random.choice(len(freqs),p=freqs)]
            if workload == "distinct":
                rnum = np.random.randint(1,maxrnum+1)
                cur.execute("select * from t where rnum = %s limit 1" % rnum)
            elif workload == "uniform": 
            #This is the point version
                randf = np.random.rand()
                cur.execute("select * from t where %s >= rnum-cnt and %s < rnum limit 1"  % (maxrnum*randf,maxrnum*randf))
            else:
                raise Exception("unkown workload type.")
            vals = np.array(cur.fetchone(),np.int32)[1:]
        if len(rcolsf) == 0:
            pvals.append(vals[:-1])
            cards.append(vals[-1])

        else:
            if not np.all(vals[len(flatten(pcols)):-1] != 0):
                continue

            if len(pcolsf) > 0:
                ppattern = pattern % tuple(vals[:-1-len(flatten(rcols))])
            else:
                ppattern = ""

            #We need to have a binary search here to get the range queries runing.
            #Step 1: Find min and max values.
            minmaxq = generateMinMaxQueryString(cur,query)
            if len(pcolsf) > 0:
                minmaxq += " WHERE %s" % ppattern
            else:
                minmaxq += " %s" % ppattern

            if minmaxq_prev != minmaxq:
                minmaxq_prev = minmaxq
                cur.execute(minmaxq)
                minmax = np.array(cur.fetchone())

                minv = np.array([minmax[i*2] for i in range(nrcols)])
                maxv = np.array([minmax[i*2+1] for i in range(nrcols)])
            
                card = vals[-1]

            center = vals[len(flatten(pcols)):-1]
            upper_range = maxv-center
            lower_range = center-minv

            #Unfortunately, we have to do the binary search. We draw a random point from the
            lower_factor = 0.0
            lower_card = 0

            upper_factor = 1.0
            upper_card = card
            
            minscale = np.random.random_sample(lower_range.shape)
            maxscale = np.random.random_sample(upper_range.shape)
            
            join_query_string = generateJoinQueryString(cur,query)
            join_query_string += (" %s" % ppattern)

            try_upper = np.rint(center + maxscale*upper_range)
            try_lower = np.rint(center - minscale*lower_range)
            join_query_string += minpattern % tuple(try_lower)
            join_query_string += maxpattern % tuple(try_upper)
            cur.execute(join_query_string)
            card = cur.fetchone()[0]
            pvals.append(list(vals[:-1-len(flatten(rcols))]))
            lvals.append(list(try_lower))
            uvals.append(list(try_upper))
            cards.append(card)

    return pvals, lvals, uvals, cards

def init(cnf,query,workload):    
    con = pg.connect(cnf)
    cur = con.cursor()

    cols = generateInvariantColumns(query)
    jcols = generateJoinColumns(query)

    pcols = generateInvariantPointColumns(query)
    pcolsf = [ (query.tables[i].tid,query.tables[i].columns[j].cid) for i in range(len(pcols)) for j in pcols[i] ]
    rcols = generateInvariantRangeColumns(query)
    rcolsf = [ (query.tables[i].tid,query.tables[i].columns[j].cid) for i in range(len(rcols)) for j in rcols[i] ]
    maxrnum = generateTemporaryHistogram(cur,query,pcols,rcols,workload)

    cur.close()
    con.commit()
    con.close()

    return maxrnum

def destroy(cnf):
    con = pg.connect(cnf)
    cur = con.cursor()
    cur.execute("drop table t;")
    cur.close()
    con.commit()
    con.close()

def generateQueryData(cnf,query,training_queries,training_fprefix,training_dprefix,test_queries,test_fprefix,test_dprefix,iterations,workload="uniform"):    
    con = pg.connect(cnf)

    cols = generateInvariantColumns(query)
    jcols = generateJoinColumns(query)

    #We need indices over all attributes. The last one's are the join attribute. This way, we can use index only plans.
    #for t, cs, jcs in zip(query.tables, cols, jcols):
    #    colsn = [t.columns[i].cid for i in cs]
    #    jcolsn = [t.columns[i].cid for i in jcs]
    #    try:
    #        create_statement = "create index mi_%s on %s (%s,%s);" % (t.tid,t.tid,",".join(colsn),",".join(jcolsn))
    #        cur.execute(create_statement)
    #    except:
    ##        pass
    pcols = generateInvariantPointColumns(query)
    pcolsf = [ (query.tables[i].tid,query.tables[i].columns[j].cid) for i in range(len(pcols)) for j in pcols[i] ]
    rcols = generateInvariantRangeColumns(query)
    rcolsf = [ (query.tables[i].tid,query.tables[i].columns[j].cid) for i in range(len(rcols)) for j in rcols[i] ]

    maxrnum = init(cnf,query,workload)
        
    con.commit()
    con.set_session('read uncommitted', readonly=True, autocommit=True)
    cur = con.cursor()
    cur.execute("set enable_seqscan=false;")

    #Numpy arrays are super convenient to translate row-wise to columnar format
    for iteration in range(0,iterations):
        if test_queries == 0:
            continue
        if not os.path.exists("%s/iteration%02d" % (test_fprefix,iteration)):
            os.makedirs("%s/iteration%02d" % (test_fprefix,iteration))
        pvals, lvals, uvals, cards = generateQueriesAndTrueSel(cur,query,cols,test_queries,workload,maxrnum)
        j = 0

        if len(pvals) >0:
            pa = np.array(pvals).astype(int) 
            for j,(t,c) in enumerate(pcolsf):
                f = open("%s/iteration%02d/%sjoin_p_%s_%s.dump" % (test_fprefix,iteration,test_dprefix,t,c),"wb")
                a = array.array("I",pa.T[j])
                a.tofile(f)
                f.close()

        if len(lvals) > 0:
            la = np.array(lvals).astype(int)
            for j,(t,c) in enumerate(rcolsf):
                f = open("%s/iteration%02d/%sjoin_l_%s_%s.dump" % (test_fprefix,iteration,test_dprefix,t,c),"wb")
                a = array.array("I",la.T[j])
                a.tofile(f)
                f.close()

        if len(uvals) >0:
            ua = np.array(uvals).astype(int) 
            for j,(t,c) in enumerate(rcolsf):
                f = open("%s/iteration%02d/%sjoin_u_%s_%s.dump" % (test_fprefix,iteration,test_dprefix,t,c),"wb")
                a = array.array("I",ua.T[j])
                a.tofile(f)
                f.close()

        cards = np.array(cards).astype(int)
        f = open("%s/iteration%02d/%sjoin_true.dump" % (test_fprefix,iteration,test_dprefix),"wb")
        a = array.array("I",cards)
        a.tofile(f)
        f.close()

    for iteration in range(0,iterations):
        if training_queries == 0:
            continue
        if not os.path.exists("%s/iteration%02d" % (training_fprefix,iteration)):
            os.makedirs("%s/iteration%02d" % (training_fprefix,iteration))
        pvals, lvals, uvals, cards = generateQueriesAndTrueSel(cur,query,cols,training_queries,workload,maxrnum)
        j = 0

        if len(pvals) >0:
            pa = np.array(pvals).astype(int) 
            for j,(t,c) in enumerate(pcolsf):
                f = open("%s/iteration%02d/%sjoin_p_%s_%s.dump" % (training_fprefix,iteration,training_dprefix,t,c),"wb")
                a = array.array("I",pa.T[j])
                a.tofile(f)
                f.close()

        if len(lvals) > 0:
            la = np.array(lvals).astype(int)
            for j,(t,c) in enumerate(rcolsf):
                f = open("%s/iteration%02d/%sjoin_l_%s_%s.dump" % (training_fprefix,iteration,training_dprefix,t,c),"wb")
                a = array.array("I",la.T[j])
                a.tofile(f)
                f.close()

        if len(uvals) >0:
            ua = np.array(uvals).astype(int) 
            for j,(t,c) in enumerate(rcolsf):
                f = open("%s/iteration%02d/%sjoin_u_%s_%s.dump" % (training_fprefix,iteration,training_dprefix,t,c),"wb")
                a = array.array("I",ua.T[j])
                a.tofile(f)
                f.close()

        cards = np.array(cards).astype(int)
        f = open("%s/iteration%02d/%sjoin_true.dump" % (training_fprefix,iteration,training_dprefix),"wb")
        a = array.array("I",cards)
        a.tofile(f)
        f.close()

    #Okay, now we need to construct the histogram for all distinct values.
    
    #for t in query.tables:
    #    drop_statement = "drop index mi_%s;" % (t.tid)
    #    cur.execute(drop_statement)

    cur.close()
    con.close()
    destroy(cnf)
