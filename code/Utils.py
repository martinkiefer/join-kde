# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:17:19 2016

@author: mkiefer
"""
import psycopg2 as pg
import json
from collections import namedtuple
import array
import sys
import numpy as np

class Uhash:
    def __init__(self,seed=None,p=4294971673):
        self.p = p
        np.random.seed(seed)
        self.a = np.random.randint(1,self.p)
        self.b = np.random.randint(1,self.p)

    def hash_int(self,x,m = None):
        if m is None:
            m = self.p-1
        return ((self.a*x+self.b) % self.p) % (m+1)  

    def hash_float(self,x):
        return self.hash_int(x,self.p-1) / float(self.p-1)  

def _json_object_hook(d): 
    return namedtuple('X', list(d.keys()))(*list(d.values()))
def json2obj(data): 
    return json.loads(data, object_hook=_json_object_hook)
    
def readCol(fname,n):
    a=array.array('I')
    f=open(fname, 'rb')
    a.fromfile(f, n)
    f.close()

    return a

def generateInvariantColumns(query):
    cols = [None]* len(query.tables)
    for j,t in enumerate(query.tables):
        cols[j] = list(range(0,len(t.columns)))
        
    for join in query.joins:
        for ic in join:
            tid,cid = ic
            cols[tid].remove(cid)
            
    return cols 

def generateInvariantPointColumns(query):
    cols = [None]* len(query.tables)
    for j,t in enumerate(query.tables):
        cols[j] = list(range(0,len(t.columns)))
        
    for join in query.joins:
        for ic in join:
            tid,cid = ic
            cols[tid].remove(cid)

    for j,t in enumerate(query.tables):
        for cid, c in enumerate(t.columns):
            if c.type != "point":
                try:
                    cols[j].remove(cid)
                except ValueError:
                    pass
            
    return cols 
    
def generateInvariantRangeColumns(query):
    cols = [None]* len(query.tables)
    for j,t in enumerate(query.tables):
        cols[j] = list(range(0,len(t.columns)))
        
    for join in query.joins:
        for ic in join:
            tid,cid = ic
            cols[tid].remove(cid)

    for j,t in enumerate(query.tables):
        for cid, c in enumerate(t.columns):
            if c.type != "range":
                try:
                    cols[j].remove(cid)
                except ValueError:
                    pass
            
    return cols 

def generateJoinColumns(query):
    cols = [None]* len(query.tables)
    for j,t in enumerate(query.tables):
        cols[j] = []
        
    for join in query.joins:
        for ic in join:
            tid,cid = ic
            cols[tid].append(cid)
            
    return cols 

def generateEquivalenceClassMap(query):
    cols = {}
       
    for jid,join in enumerate(query.joins):
        for ic in join:
            tid,cid = ic
            cols[(tid,cid)] = jid
            
    return cols 

    
#Keep in mind that tuples might be returned in reverse order if restrict is provided
def generateJoinPairs(query,restrict=None):
    cols = []
    ctr = 0
    for j,join in enumerate(query.joins):
        for i in range(0,len(join)-1):
            p1 = join[i]
            p2 = join[i+1]

            if restrict == None: 
                cols.append((ctr,p1,p2))
            elif p1[0]==restrict:
                cols.append((ctr,p1,p2))
            elif p2[0]==restrict:
                cols.append((ctr,p2,p1))            
            ctr += 1
    return cols
    
    
def flatten(l):
    return [x for sl in l for x in sl]
    
#Collect table sizes and number of distinct values
def retreiveTableStatistics(cnf,query):
    dv = {}
    ts = {}
    
    con = pg.connect(cnf)
    cur = con.cursor()
    
    for j,t in enumerate(query.tables):
        sql_query = "select count(*)" 
        for k,c in enumerate(t.columns):
            sql_query = "%s, count(distinct( %s ))" % (sql_query,c.cid) 
        sql_query = "%s from %s" % (sql_query,t.tid)
        
        cur.execute(sql_query)
        tup = cur.fetchone()

        ts[j] = tup[0]
        dv[j] = tup[1:]
    cur.close()
    con.close()
    return (ts,dv)

def retreiveJoinStatistics(cnf,query):      
    cols = generateInvariantColumns(query)
    first = True
    query_select = "select count(*),"
    j = 1
    for i,col in enumerate(cols):
        for c in col:
            if not first:
                query_select += ", "
            query_select += "count(distinct(%s.%s))" % (query.tables[i].tid,query.tables[i].columns[c].cid)
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
   
    con = pg.connect(cnf)
    cur = con.cursor()
    cur.execute(query_select)
    tup = cur.fetchone()
    js = tup[0]
    dvs = tup[1:]
    cur.close()
    con.close()
    return js,dvs
    
def generateFileCheckFunction(f):
    print("""
inline bool fexists(const char* name) {
    std::ifstream f(name);
    return f.good();
}
""", file=f)   

def generateDoubleDumper(f):
    print("""
void ddump(double* array, size_t n, const char* name) {
    FILE* f = fopen(name,"wb");
    fwrite(array, sizeof(double), n, f);
    fclose(f);
}
""", file=f)   

def generateGPUVectorConverterFunction(f):
    print("""
template <typename T>
compute::vector<T> toGPUVector(T* arr, size_t n, compute::context &context, compute::command_queue &queue){
    compute::vector<T> vec (n, context);
    compute::copy(
            arr, arr+n, vec.begin(), queue
    );
   return vec;
}  
""", file=f)   

def generateUintFileReaderFunction(f):    
    print("""unsigned int* readUArrayFromFile(const char* filename, size_t * filesize = NULL){
    FILE *f1 = fopen(filename, "rb");
    assert(f1 != NULL);
    fseek(f1, 0, SEEK_END);
    size_t fsize1 = ftell(f1);
    if(filesize) *filesize=fsize1;
    fseek(f1, 0, SEEK_SET);
    unsigned int* tab1 = (unsigned int*) malloc(fsize1);
    size_t x = fread(tab1, fsize1, 1, f1);
    fclose(f1);

    return tab1;
}
""", file=f)

def generateDoubleFileReaderFunction(f):    
    print("""   
double* readDArrayFromFile(const char* filename){
    FILE *f1 = fopen(filename, "rb");
    assert(f1 != NULL);
    fseek(f1, 0, SEEK_END);
    size_t fsize1 = ftell(f1);
    fseek(f1, 0, SEEK_SET);
    double* tab1 = (double*) malloc(fsize1);
    size_t x = fread(tab1, fsize1, 1, f1);
    fclose(f1);
    
    return tab1;
}
""", file=f)

def generateScottBWFunction(f):
    print("""
double scott_bw(unsigned int* sample, unsigned int sample_size, unsigned int d){
    double mean = 0.0;
    double sdev = 0.0;
    unsigned int i = 0;

    for(i = 0; i < sample_size; i++){
      mean += sample[i];
    }
    mean /= sample_size;

    for(i = 0; i < sample_size; i++){
      sdev += (sample[i]-mean)*(sample[i]-mean);
    }

    sdev = sqrt(sdev/(sample_size-1));
    if(sdev <= 10e-10) sdev = 1.0;

    return pow((double)sample_size,-1.0/(d+4))*sdev;
}
""", file=f) 

def printResultString(modelsize,est,trues,iteration):
    print("%s,%s,%f,%f,%f,%f,%f,%f" % (iteration,modelsize,est,trues,(est-trues)*(est-trues),abs(est-trues),abs(est-trues)/trues,max(max(est,1.0)/max(trues,1.0),max(trues,1.0)/max(est,1.0))))
    sys.stdout.flush()
    
def generateRoundMethod(f):
    print("""
size_t roundUp(size_t numToRound, size_t multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}
    """, file=f)

def printObjectiveLine(f,modelsize):
    print("        if(est < 1.0) est = 1.0;", file=f)
    print("        std::cout << p->iteration << \",\" << %s << \",\" << est << \",\" << trues << \",\" << (fmax(est,0)-trues)*(fmax(est,0)-trues) << \",\" << fabs(fmax(est,0)-trues) << \",\" << fabs(fmax(est,0)-trues)/trues << \",\" << fmax(fmax(trues,1.0)/fmax(est,1.0),fmax(est,1.0)/fmax(trues,1.0)) << \",\" <<  std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << std::endl;" % modelsize, file=f)
