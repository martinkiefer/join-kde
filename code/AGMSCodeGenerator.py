 #Code generator for gauss kernel
import Utils
from Utils import generateRoundMethod

def generatePreamble(f):
    print >>f, "#pragma OPENCL EXTENSION cl_khr_fp64 : enable "
    print >>f, "typedef double T;"
    print >>f, """
int parity(unsigned int x) {
   unsigned int y;
   y = x ^ (x >> 1);
   y = y ^ (y >> 2);
   y = y ^ (y >> 4);
   y = y ^ (y >> 8);
   y = y ^ (y >>16);
   return y & 1;
}

int nonlinear_h(unsigned int x) {
    return parity((x >> 0) | (x >> 1));
}

int is_set(unsigned int x, unsigned int pos) {
    return (x >> pos) & 1;
}


int ech3(unsigned int v, unsigned int seed, int sbit){
    //First we compute the bitwise AND between the seed and the value
    //Aaaand here comes the parity
    int res = (parity(v & seed) != nonlinear_h(v)) != sbit ; 
    return 2*res-1; 
}

int range_ech3(unsigned int u, unsigned int l, unsigned int seed, int sbit){
    int ctr = 0;   
    for(unsigned int i = l; i <= u; i++) ctr +=ech3(i, seed, sbit);
    return ctr;
}
    """
    
def generateCIncludes(f):
    print >>f, """
    
#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <cmath>

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace compute = boost::compute;
"""

def generateParameterArray(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    jpairs = Utils.generateJoinPairs(query)
    print >>f, """
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
"""
    print >>f, "    unsigned int iteration;"
    print >>f, "    size_t skn;"
    for j,t in enumerate(query.tables):
        print >>f, "    compute::vector<long> sk_t%s;" % (j) 
        print >>f, "    unsigned int ts%s;" % (j) 
        print >>f, "    compute::kernel t%s_construct_sketch;" % j
        for k,c in enumerate(t.columns):
            print >>f, "    compute::vector<unsigned int> t%s_c%s;" % (j,k)
            print >>f, "    compute::vector<unsigned int> t%s_c%s_lseed;" % (j,k)
            print >>f, "    compute::vector<unsigned int> t%s_c%s_sseed;" % (j,k)
        print >>f
        
    for j,p1,p2 in jpairs:
        print >>f, "    compute::vector<unsigned int> j%s_t%s_c%s_t%s_c%s_lseed;" % (j,p1[0],p1[1],p2[0],p2[1])
        print >>f, "    compute::vector<unsigned int> j%s_t%s_c%s_t%s_c%s_sseed;" % (j,p1[0],p1[1],p2[0],p2[1])
    print >>f, "    compute::kernel multiply_sketches;"
    
    #Training
    print >>f, "    compute::vector<double> estimates;"
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        for j in indices:
            if query.tables[i].columns[j].type == "point":
                print >>f, "    unsigned int* j_p_t%s_c%s;" % (i,j)
            elif query.tables[i].columns[j].type == "range":
                print >>f, "    unsigned int* j_u_t%s_c%s;" % (i,j)
                print >>f, "    unsigned int* j_l_t%s_c%s;" % (i,j)
            else:
                raise Exception("Unknown column type.")

    print >>f, "    unsigned int* j_test_cardinality;"                
    print >>f, """
} parameters;
"""   

def generateSketchConstructionCCode(f,query,ts,local_size=64,cu_factor=2048):
    icols = Utils.generateInvariantColumns(query) 
    jpairs = Utils.generateJoinPairs(query)
    
    print >>f, "void sketch_contruction(parameters* p){"
    print >>f, "    size_t local = %s;" % local_size 
    print >>f, "    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((p->skn-1)/local+1)*local);" % cu_factor
    for tid, tab in enumerate(query.tables):
        cs = ",".join(map(lambda x : "p->t%s_c%s" % (tid,x),range(0,len(tab.columns))))            
        cseeds = ",".join(map(lambda x : "p->t%s_c%s_lseed, p->t%s_c%s_sseed" % (tid,x,tid,x), icols[tid]))
        jseeds = ",".join(map(lambda y : "p->j%s_t%s_c%s_t%s_c%s_lseed, p->j%s_t%s_c%s_t%s_c%s_sseed" % (y[0],y[1][0],y[1][1],y[2][0],y[2][1],y[0],y[1][0],y[1][1],y[2][0],y[2][1]) ,filter(lambda x : x[1][0] == tid or x[2][0] == tid,jpairs)))
        

        print >>f, "    p->t%s_construct_sketch.set_args((unsigned int) p->skn,%s,%s,%s,p->sk_t%s);" % (tid,cs,cseeds,jseeds,tid)
        print >>f, "    boost::compute::event ev%s = p->queue.enqueue_nd_range_kernel(p->t%s_construct_sketch,1,NULL,&global,&local);" % (tid,tid)
    
    for tid, tab in enumerate(query.tables):   
        print >>f, "    ev%s.wait();" % tid
    print >>f, "}"
    
def generateEstimateCCode(f,query,cu_factor=2048):
    icols = Utils.generateInvariantColumns(query) 
    jpairs = Utils.generateJoinPairs(query)
    
    print >>f, "double estimate(parameters* p",
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print >>f, ", unsigned int t%s_c%s" % (i,j),                 
            elif query.tables[i].columns[j].type == "range":
                print >>f, ", unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (i,j,i,j),                 
            else:
                raise Exception("unknown column type.")

    print >>f, "){"
    #Aaaaaaaaaand construct the estimation kernel
    print >>f, "    size_t local = 64;"
    print >>f, "    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*2048 , ((p->skn-1)/local+1)*local);"
    print >>f, "    p->multiply_sketches.set_args(",
    for tid, tab in enumerate(query.tables):
        print >>f, "p->sk_t%s, " % tid,
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print >>f, "t%s_c%s, p->t%s_c%s_lseed, p->t%s_c%s_sseed," % (i,j,i,j,i,j),    
            elif query.tables[i].columns[j].type == "range":
                print >>f, "u_t%s_c%s, l_t%s_c%s, p->t%s_c%s_lseed, p->t%s_c%s_sseed," % (i,j,i,j,i,j,i,j),    
            else:
                raise Exception("unknown column type.")

    print >>f, "p->estimates, (unsigned int) p->skn);"
    print >>f, "    boost::compute::event ev = p->queue.enqueue_nd_range_kernel(p->multiply_sketches, 1, NULL, &global, &local);"
    print >>f, "    ev.wait();"
    print >>f, "    double est = 0.0;"
    print >>f, "    boost::compute::reduce(p->estimates.begin(), p->estimates.end(), &est, p->queue);"
    print >>f, "    p->queue.finish();"
    print >>f, "    return est / p->skn;"    
    print >>f, "}"    
    
def generateTestWrapper(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    #print >>f, "double join_test(unsigned n, const double* bw, double* grad, void* f_data){"
    print >>f, "double join_test(parameters* p){"    
    #print >>f, "    parameters* p = (parameters*) f_data;"
    print >>f, "    double objective = 0.0;"
    print >>f, "    double trues = 0.0;"
    print >>f, "    double est = 0.0;"
    print >>f, "    int first = 1;"
    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.test
    print >>f, "        auto begin = std::chrono::high_resolution_clock::now();"
    print >>f, "        if(first ",
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions
        if len(indices) != 0:
            for j in indices:
                if query.tables[i].columns[j].type == "point":
                    print >>f, "|| p->j_p_t%s_c%s[i] != p->j_p_t%s_c%s[i-1] " % (i,j,i,j),
                elif query.tables[i].columns[j].type == "range":
                    print >>f, "|| p->j_u_t%s_c%s[i] != p->j_u_t%s_c%s[i-1] " % (i,j,i,j),
                    print >>f, "|| p->j_l_t%s_c%s[i] != p->j_l_t%s_c%s[i-1] " % (i,j,i,j),
                else:
                    raise Exception("Unknown column type.")
    print >>f, "){"
    if hasattr(estimator, 'look_behind'):
        if estimator.look_behind:
            print >> f, "            first = 0;"
    else:
        print >>f, "            first = 0;"
    print >>f, "            est = estimate(p",
                                                            
    for i,indices in enumerate(cols):
        for j in indices:
            if query.tables[i].columns[j].type == "point":
                print >>f, ", p->j_p_t%s_c%s[i]" % (i,j), 
            elif query.tables[i].columns[j].type == "range":
                print >>f, ", p->j_u_t%s_c%s[i]" % (i,j), 
                print >>f, ", p->j_l_t%s_c%s[i]" % (i,j), 
            else:
                raise Exception("Unkown column type.")
    print >>f, ");"
    print >>f, "        }"
    print >>f, "        auto end = std::chrono::high_resolution_clock::now();"
    print >>f, "        trues = p->j_test_cardinality[i];"
    print >>f, "        objective += (est-trues)*(est-trues);" 
    Utils.printObjectiveLine(f,"p->skn")
    print >>f, "    }"
    print >>f, "    return objective/%s;" % estimator.test
    print >>f, "}"

def generateSketchConstructionCode(f,query,ts,local_size=32):
    icols = Utils.generateInvariantColumns(query)    
    for tid,tab in enumerate(query.tables):
        pairs = Utils.generateJoinPairs(query,tid)
        print >>f, "__kernel void t%s_construct_sketch(" % tid
        #Get column buffers
        print >> f, "    unsigned int skn,"
        for cid, cols in enumerate(tab.columns):
            print >>f, "    __global unsigned int* c%s," % cid
        #Get seeds for invariant columns
        for cid in icols[tid]:
            print >>f, "    __global unsigned int* c%s_ls, __global unsigned int* c%s_ss," % (cid,cid)
        for j,p1,p2 in pairs:
            print >>f, "    __global unsigned int* j%s_ls, __global unsigned int* j%s_ss," % (j,j)
        print >>f, "    __global long* sketches) {"
        
        for cid, cols in enumerate(tab.columns):
            print >>f, "    __local unsigned int cache_c%s[%s];" % (cid,local_size)
        print >>f, "    for(unsigned int offset = 0; offset < skn; offset += get_global_size(0)){" 
        print >>f, "           long counter = 0;"
        for cid in icols[tid]:
            print >>f, "           unsigned int my_c%s_ls = (get_global_id(0)+offset < skn) ? c%s_ls[get_global_id(0)+offset] : 0; int my_c%s_ss = (get_global_id(0)+offset < skn) ? is_set(c%s_ss[(get_global_id(0)+offset)/32],(get_global_id(0)+offset) %% 32) : 0;" % (cid,cid,cid,cid)
        for j,p1,p2 in pairs:
            print >>f, "           unsigned int my_j%s_ls = (get_global_id(0)+offset < skn) ? j%s_ls[get_global_id(0)+offset] : 0; int my_j%s_ss = (get_global_id(0)+offset < skn) ? is_set(j%s_ss[(get_global_id(0)+offset)/32],(get_global_id(0)+offset)%% 32) : 0;" % (j,j,j,j)
    
        print >>f, "        for(unsigned int i = 0; i < %s; i += get_local_size(0)){" % (ts[tid])
        print >>f, "            barrier(CLK_LOCAL_MEM_FENCE);"
        print >>f, "            if(i + get_local_id(0) < %s) {" % (ts[tid])
        for cid, cols in enumerate(tab.columns):
            print >>f, "                cache_c%s[get_local_id(0)] = c%s[i+get_local_id(0)];" % (cid,cid)
        print >>f, "            }"
        print >>f, "            barrier(CLK_LOCAL_MEM_FENCE);"
        print >>f, "            if(get_global_id(0)+offset >= skn) continue;"
        print >>f, "            for(unsigned int j = 0; j < get_local_size(0) && i+j < %s; j++){" % ts[tid]
        print >>f, "                counter += %s * %s;" % (" * ".join(map(lambda x : "ech3(cache_c%s[j],my_c%s_ls,my_c%s_ss)" % (x,x,x), icols[tid]))," * ".join(map(lambda x : "ech3(cache_c%s[j],my_j%s_ls,my_j%s_ss)" % (x[1][1],x[0],x[0]), pairs)))
        print >>f, "            }"
        print >>f, "        }"
        print >>f, "        if(get_global_id(0)+offset < skn) sketches[get_global_id(0)+offset] = counter;"
        print >>f, "    }"    
        print >>f, "}"    
        print >>f
            
        
def generateSketchMultiplyCode(f,query):  
    icols = Utils.generateInvariantColumns(query)   
             
    print >>f, "__kernel void multiply_sketches("

    for tid,tab in enumerate(query.tables):
        print >>f, "    __global long* t%s_sketches," % tid
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print >>f, "    unsigned int t%s_c%s, __global unsigned int* t%s_c%s_ls, __global unsigned int* t%s_c%s_ss," % (i,j,i,j,i,j)
            elif query.tables[i].columns[j].type == "range":
                print >>f, "    unsigned int u_t%s_c%s, unsigned int l_t%s_c%s, __global unsigned int* t%s_c%s_ls, __global unsigned int* t%s_c%s_ss," % (i,j,i,j,i,j,i,j)
            else:
                raise Exception("Unknown column type.")

    print >>f, "    __global double* estimates, unsigned int skn) {"
    print >> f, "    for(unsigned int offset = 0; offset < skn; offset += get_global_size(0)){"
    print >> f, "        if(get_global_id(0) + offset < skn){"
    for i,col in enumerate(icols):
        for j in col:
            print >>f, "            unsigned int my_t%s_c%s_ls = t%s_c%s_ls[get_global_id(0)+offset]; int my_t%s_c%s_ss = is_set(t%s_c%s_ss[(get_global_id(0)+offset)/32],(get_global_id(0)+offset) %% 32);" % (i,j,i,j,i,j,i,j)   
    
    print >>f, "            estimates[get_global_id(0)+offset] = %s " % (" * ".join(map(lambda x : "t%s_sketches[get_global_id(0)+offset]" % x,range(0,len(query.tables))))),
    
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print >>f, "* ech3(t%s_c%s, my_t%s_c%s_ls, my_t%s_c%s_ss)" % (i,j,i,j,i,j),
            elif query.tables[i].columns[j].type == "range":
                print >>f, "* range_ech3(u_t%s_c%s, l_t%s_c%s, my_t%s_c%s_ls, my_t%s_c%s_ss)" % (i,j,i,j,i,j,i,j),
            else:
                raise Exception("Invalid column type.")
                
    print >>f, ";"
    print >>f, "        }"
    print >>f, "    }"
    print >>f, "}"
    
def generateAGMSCode(i,query,estimator,stats,cu_factor):
    ts, dv = stats
    local_size = 64
    
    icols = Utils.generateInvariantColumns(query)    
    #jcols = Utils.generateInvariantColumns(query)
    jpairs = Utils.generateJoinPairs(query)
    
    with open("./%s_kernels.cl" % i,'w') as cf:
        generatePreamble(cf)
        generateSketchConstructionCode(cf,query,ts,local_size)
        
        generateSketchMultiplyCode(cf,query)
    
    with open("./%s_AGMS.cpp" % i,'w') as cf:
        generateCIncludes(cf)

        generateRoundMethod(cf)
        generateParameterArray(cf,query,estimator)
        Utils.generateGPUVectorConverterFunction(cf)
        Utils.generateUintFileReaderFunction(cf)
        generateEstimateCCode(cf,query,cu_factor)
        generateSketchConstructionCCode(cf,query,ts,local_size,cu_factor)  
        generateTestWrapper(cf,query,estimator)
            
        print >>cf, """
int main( int argc, const char* argv[] ){
    parameters p;
    compute::device device = compute::system::default_device();
    p.ctx = compute::context(device);
    p.queue=compute::command_queue(p.ctx, device);
"""
        print >>cf, """
    std::ifstream t("./%s_kernels.cl");
    t.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    std::string str((std::istreambuf_iterator<char>(t)),
    std::istreambuf_iterator<char>());
""" % i
        print >>cf, """
    compute::program pr = compute::program::create_with_source(str,p.ctx);
    try{
        std::ostringstream oss;
        pr.build(oss.str());
    } catch(const std::exception& ex){
        std::cout << pr.build_log() << std::endl;
    }
        """
        print >>cf, "    p.iteration = atoi(argv[2]);"
        print >>cf, "    std::stringstream iteration_stream;"
        print >>cf, "    iteration_stream << \"./iteration\" << std::setw(2) << std::setfill('0') << argv[2];"
        print >>cf, "    p.skn = (unsigned int) atoll(argv[1]);"
        print >>cf, "    p.estimates = compute::vector<double>(p.skn, p.ctx);"
        for j,t in enumerate(query.tables):
            print >>cf, "    p.ts%s= %s;" % (j,ts[j])
            print >>cf, "    p.sk_t%s = compute::vector<long>(p.skn, p.ctx);" % j
            for k,c in enumerate(t.columns):
                print >>cf, "    std::stringstream t%s_c%s_stream ;" % (j,k)
                print >>cf, "    t%s_c%s_stream <<  \"./table_%s_%s.dump\";" % (j,k,t.tid,c.cid) 
                print >>cf, "    std::string t%s_c%s_string = t%s_c%s_stream.str();" % (j,k,j,k)
                print >>cf, "    unsigned int* t%s_c%s = readUArrayFromFile(t%s_c%s_string.c_str());" % (j,k,j,k)
                print >>cf, "    p.t%s_c%s = toGPUVector(t%s_c%s, p.ts%s, p.ctx, p.queue);" % (j,k,j,k,j)

        print >>cf, "    boost::random::mt19937 gen;"
        
        for x, cols in enumerate(icols):
            for y in cols: 
                print >>cf, "    unsigned int* t%s_c%s_lseed =  (unsigned int*) malloc(sizeof(unsigned int)*p.skn);" % (x,y)
                print >>cf, "    unsigned int* t%s_c%s_sseed =  (unsigned int*) malloc(((p.skn-1)/(sizeof(unsigned int)*8) +1)*sizeof(8));" % (x,y)
                
                print >>cf, "    for(unsigned int i = 0; i < p.skn;  i++ ){"
                print >>cf, "       t%s_c%s_lseed[i] = gen();" % (x,y)
                print >>cf, "    }"
                
                print >>cf, "    for(unsigned int i = 0; i < ((p.skn-1)/(sizeof(unsigned int)*8) +1);  i++ ){" 
                print >>cf, "       t%s_c%s_sseed[i] = gen();"  % (x,y)
                print >>cf, "    }"

                print >>cf, "    p.t%s_c%s_lseed =  toGPUVector(t%s_c%s_lseed, p.skn, p.ctx, p.queue);" % (x,y,x,y)
                print >>cf, "    p.t%s_c%s_sseed =  toGPUVector(t%s_c%s_sseed, ((p.skn-1)/(sizeof(unsigned int)*8) +1), p.ctx, p.queue);" % (x,y,x,y)                
                
        
        for j,p1,p2 in jpairs:
            print >>cf, "    unsigned int* j%s_t%s_c%s_t%s_c%s_lseed =  (unsigned int*) malloc(sizeof(unsigned int)*p.skn);" % (j,p1[0],p1[1],p2[0],p2[1])
            print >>cf, "    unsigned int* j%s_t%s_c%s_t%s_c%s_sseed =  (unsigned int*) malloc(((p.skn-1)/(sizeof(unsigned int)*8) +1)*sizeof(unsigned int));" % (j,p1[0],p1[1],p2[0],p2[1])
            
            print >>cf, "    for(unsigned int i = 0; i < p.skn;  i++ ){"
            print >>cf, "       j%s_t%s_c%s_t%s_c%s_lseed[i] = gen();" % (j,p1[0],p1[1],p2[0],p2[1])
            print >>cf, "    }"
            
            print >>cf, "    for(unsigned int i = 0; i < ((p.skn-1)/(sizeof(unsigned int)*8) +1);  i++ ){" 
            print >>cf, "       j%s_t%s_c%s_t%s_c%s_sseed[i] = gen();"  % (j,p1[0],p1[1],p2[0],p2[1])
            print >>cf, "    }"
 
            print >>cf, "    p.j%s_t%s_c%s_t%s_c%s_lseed = toGPUVector(j%s_t%s_c%s_t%s_c%s_lseed, p.skn, p.ctx, p.queue);" % (j,p1[0],p1[1],p2[0],p2[1],j,p1[0],p1[1],p2[0],p2[1])
            print >>cf, "    p.j%s_t%s_c%s_t%s_c%s_sseed = toGPUVector(j%s_t%s_c%s_t%s_c%s_sseed, ((p.skn-1)/(sizeof(unsigned int)*8) +1), p.ctx, p.queue); " % (j,p1[0],p1[1],p2[0],p2[1],j,p1[0],p1[1],p2[0],p2[1])    
            
        for j,t in enumerate(query.tables):    
            print >>cf, "    p.t%s_construct_sketch = pr.create_kernel(\"t%s_construct_sketch\");" % (j,j) 
        print >>cf, "    p.multiply_sketches = pr.create_kernel(\"multiply_sketches\");"
        print >>cf, "    sketch_contruction(&p);"
        
        print >>cf, "    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";"
        print >>cf, "    p.j_test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());"
        
        for i,indices in enumerate(icols):
            for j in indices:
                if query.tables[i].columns[j].type == "point": 
                    print >>cf, "    std::string j_p_t%s_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                    print >>cf, "    p.j_p_t%s_c%s = readUArrayFromFile(j_p_t%s_c%s_string.c_str());" % (i,j,i,j)
                elif query.tables[i].columns[j].type == "range": 
                    print >>cf, "    std::string j_u_t%s_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                    print >>cf, "    std::string j_l_t%s_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                    print >>cf, "    p.j_u_t%s_c%s = readUArrayFromFile(j_u_t%s_c%s_string.c_str());" % (i,j,i,j)
                    print >>cf, "    p.j_l_t%s_c%s = readUArrayFromFile(j_l_t%s_c%s_string.c_str());" % (i,j,i,j)
                else:
                    raise Exception("Unknown column type.")
        
        print >>cf
        print >>cf, "    join_test(&p);"
        print >>cf, "}"
