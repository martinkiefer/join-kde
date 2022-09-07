 #Code generator for gauss kernel
import Utils
from Utils import generateRoundMethod

def generatePreamble(f):
    print("#pragma OPENCL EXTENSION cl_khr_fp64 : enable ", file=f)
    print("typedef double T;", file=f)
    print("""
int parity(unsigned int x) {
    return popcount(x) & 1;
}

int nonlinear_h(unsigned int x) {
    return parity(((x >> 0) & (x >> 1)) & 0x55555555);
}

unsigned int is_set(unsigned int x, unsigned int pos) {
    return (x >> pos) & 1;
}


int ech3(unsigned int v, unsigned int seed, unsigned int sbit){
    //First we compute the bitwise AND between the seed and the value
    //Aaaand here comes the parity
    int res = parity(v & seed) ^ nonlinear_h(v) ^ sbit ; 
    return 2*res-1; 
}

int range_ech3(unsigned int u, unsigned int l, unsigned int seed, unsigned int sbit){
    int ctr= 0;
    u++;
    while(l < u){
        unsigned int j = min(ctz(l) >> 1, 16-clz(u-l)/2-1);
        unsigned int q = l >> 2*j;
        unsigned int ut = (q+1) << 2*j;
        int zeros = 2*parity(seed & (seed >> 1) & ((0x55555555 >> (32-j*2))*(j!=0))) - 1;
        ctr += ech3(l, seed, sbit) * -zeros * (1 << j);
        l = ut;
    }
    return ctr;
}
    """, file=f)
    
def generateCIncludes(f):
    print("""
    
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
""", file=f)

def generateParameterArray(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    jpairs = Utils.generateJoinPairs(query)
    print("""
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
""", file=f)
    print("    unsigned int iteration;", file=f)
    print("    size_t skn;", file=f)
    for j,t in enumerate(query.tables):
        print("    compute::vector<long> sk_t%s;" % (j), file=f) 
        print("    unsigned int ts%s;" % (j), file=f) 
        print("    compute::kernel t%s_construct_sketch;" % j, file=f)
        for k,c in enumerate(t.columns):
            print("    compute::vector<unsigned int> t%s_c%s;" % (j,k), file=f)
            print("    compute::vector<unsigned int> t%s_c%s_lseed;" % (j,k), file=f)
            print("    compute::vector<unsigned int> t%s_c%s_sseed;" % (j,k), file=f)
        print(file=f)
        
    for j,p1,p2 in jpairs:
        print("    compute::vector<unsigned int> j%s_t%s_c%s_t%s_c%s_lseed;" % (j,p1[0],p1[1],p2[0],p2[1]), file=f)
        print("    compute::vector<unsigned int> j%s_t%s_c%s_t%s_c%s_sseed;" % (j,p1[0],p1[1],p2[0],p2[1]), file=f)
    print("    compute::kernel multiply_sketches;", file=f)
    
    #Training
    print("    compute::vector<double> estimates;", file=f)
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        for j in indices:
            if query.tables[i].columns[j].type == "point":
                print("    unsigned int* j_p_t%s_c%s;" % (i,j), file=f)
            elif query.tables[i].columns[j].type == "range":
                print("    unsigned int* j_u_t%s_c%s;" % (i,j), file=f)
                print("    unsigned int* j_l_t%s_c%s;" % (i,j), file=f)
            else:
                raise Exception("Unknown column type.")

    print("    unsigned int* j_test_cardinality;", file=f)                
    print("""
} parameters;
""", file=f)   

def generateSketchConstructionCCode(f,query,ts,local_size=64,cu_factor=2048):
    icols = Utils.generateInvariantColumns(query) 
    jpairs = Utils.generateJoinPairs(query)
    
    print("void sketch_contruction(parameters* p){", file=f)
    print("    size_t local = %s;" % local_size, file=f) 
    print("    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((p->skn-1)/local+1)*local);" % cu_factor, file=f)
    for tid, tab in enumerate(query.tables):
        cs = ",".join(["p->t%s_c%s" % (tid,x) for x in range(0,len(tab.columns))])            
        cseeds = ",".join(["p->t%s_c%s_lseed, p->t%s_c%s_sseed" % (tid,x,tid,x) for x in icols[tid]])
        jseeds = ",".join(["p->j%s_t%s_c%s_t%s_c%s_lseed, p->j%s_t%s_c%s_t%s_c%s_sseed" % (y[0],y[1][0],y[1][1],y[2][0],y[2][1],y[0],y[1][0],y[1][1],y[2][0],y[2][1]) for y in [x for x in jpairs if x[1][0] == tid or x[2][0] == tid]])
        

        print("    p->t%s_construct_sketch.set_args((unsigned int) p->skn,%s,%s,%s,p->sk_t%s);" % (tid,cs,cseeds,jseeds,tid), file=f)
        print("    boost::compute::event ev%s = p->queue.enqueue_nd_range_kernel(p->t%s_construct_sketch,1,NULL,&global,&local);" % (tid,tid), file=f)
    
    for tid, tab in enumerate(query.tables):   
        print("    ev%s.wait();" % tid, file=f)
    print("}", file=f)
    
def generateEstimateCCode(f,query,cu_factor=2048):
    icols = Utils.generateInvariantColumns(query) 
    jpairs = Utils.generateJoinPairs(query)
    
    print("double estimate(parameters* p", end=' ', file=f)
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print(", unsigned int t%s_c%s" % (i,j), end=' ', file=f)                 
            elif query.tables[i].columns[j].type == "range":
                print(", unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (i,j,i,j), end=' ', file=f)                 
            else:
                raise Exception("unknown column type.")

    print("){", file=f)
    #Aaaaaaaaaand construct the estimation kernel
    print("    size_t local = 64;", file=f)
    print("    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*2048 , ((p->skn-1)/local+1)*local);", file=f)
    print("    p->multiply_sketches.set_args(", end=' ', file=f)
    for tid, tab in enumerate(query.tables):
        print("p->sk_t%s, " % tid, end=' ', file=f)
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print("t%s_c%s, p->t%s_c%s_lseed, p->t%s_c%s_sseed," % (i,j,i,j,i,j), end=' ', file=f)    
            elif query.tables[i].columns[j].type == "range":
                print("u_t%s_c%s, l_t%s_c%s, p->t%s_c%s_lseed, p->t%s_c%s_sseed," % (i,j,i,j,i,j,i,j), end=' ', file=f)    
            else:
                raise Exception("unknown column type.")

    print("p->estimates, (unsigned int) p->skn);", file=f)
    print("    boost::compute::event ev = p->queue.enqueue_nd_range_kernel(p->multiply_sketches, 1, NULL, &global, &local);", file=f)
    print("    ev.wait();", file=f)
    print("    double est = 0.0;", file=f)
    print("    boost::compute::reduce(p->estimates.begin(), p->estimates.end(), &est, p->queue);", file=f)
    print("    p->queue.finish();", file=f)
    print("    return est / p->skn;", file=f)    
    print("}", file=f)    
    
def generateTestWrapper(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    #print >>f, "double join_test(unsigned n, const double* bw, double* grad, void* f_data){"
    print("double join_test(parameters* p){", file=f)    
    #print >>f, "    parameters* p = (parameters*) f_data;"
    print("    double objective = 0.0;", file=f)
    print("    double trues = 0.0;", file=f)
    print("    double est = 0.0;", file=f)
    print("    int first = 1;", file=f)
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.test, file=f)
    print("        auto begin = std::chrono::high_resolution_clock::now();", file=f)
    print("            est = estimate(p", end=' ', file=f)
                                                            
    for i,indices in enumerate(cols):
        for j in indices:
            if query.tables[i].columns[j].type == "point":
                print(", p->j_p_t%s_c%s[i]" % (i,j), end=' ', file=f) 
            elif query.tables[i].columns[j].type == "range":
                print(", p->j_u_t%s_c%s[i]" % (i,j), end=' ', file=f) 
                print(", p->j_l_t%s_c%s[i]" % (i,j), end=' ', file=f) 
            else:
                raise Exception("Unkown column type.")
    print(");", file=f)
    print("        auto end = std::chrono::high_resolution_clock::now();", file=f)
    print("        trues = p->j_test_cardinality[i];", file=f)
    print("        objective += (est-trues)*(est-trues);", file=f) 
    Utils.printObjectiveLine(f,"p->skn")
    print("    }", file=f)
    print("    return objective/%s;" % estimator.test, file=f)
    print("}", file=f)

def generateSketchConstructionCode(f,query,ts,local_size=32):
    icols = Utils.generateInvariantColumns(query)    
    for tid,tab in enumerate(query.tables):
        pairs = Utils.generateJoinPairs(query,tid)
        print("__kernel void t%s_construct_sketch(" % tid, file=f)
        #Get column buffers
        print("    unsigned int skn,", file=f)
        for cid, cols in enumerate(tab.columns):
            print("    __global unsigned int* c%s," % cid, file=f)
        #Get seeds for invariant columns
        for cid in icols[tid]:
            print("    __global unsigned int* c%s_ls, __global unsigned int* c%s_ss," % (cid,cid), file=f)
        for j,p1,p2 in pairs:
            print("    __global unsigned int* j%s_ls, __global unsigned int* j%s_ss," % (j,j), file=f)
        print("    __global long* sketches) {", file=f)
        
        print("    for(unsigned int offset = 0; offset < skn; offset += get_global_size(0)){", file=f) 
        print("            if(get_global_id(0)+offset >= skn) continue;", file=f)
        print("           long counter = 0;", file=f)
        for cid in icols[tid]:
            print("           unsigned int my_c%s_ls = c%s_ls[get_global_id(0)+offset]; int my_c%s_ss = is_set(c%s_ss[(get_global_id(0)+offset)/32],(get_global_id(0)+offset) %% 32);" % (cid,cid,cid,cid), file=f)
        for j,p1,p2 in pairs:
            print("           unsigned int my_j%s_ls = j%s_ls[get_global_id(0)+offset]; int my_j%s_ss = is_set(j%s_ss[(get_global_id(0)+offset)/32],(get_global_id(0)+offset)%% 32);" % (j,j,j,j), file=f)
    
        print("        for(unsigned int i = 0; i < %s; i += 1){" % (ts[tid]), file=f)
        print("                counter += %s * %s;" % (" * ".join(["ech3(c%s[i],my_c%s_ls,my_c%s_ss)" % (x,x,x) for x in icols[tid]])," * ".join(["ech3(c%s[i],my_j%s_ls,my_j%s_ss)" % (x[1][1],x[0],x[0]) for x in pairs])), file=f)
        print("        }", file=f)
        print("        sketches[get_global_id(0)+offset] = counter;", file=f)
        print("    }", file=f)    
        print("}", file=f)    
        print(file=f)
            
        
def generateSketchMultiplyCode(f,query):  
    icols = Utils.generateInvariantColumns(query)   
             
    print("__kernel void multiply_sketches(", file=f)

    for tid,tab in enumerate(query.tables):
        print("    __global long* t%s_sketches," % tid, file=f)
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print("    unsigned int t%s_c%s, __global unsigned int* t%s_c%s_ls, __global unsigned int* t%s_c%s_ss," % (i,j,i,j,i,j), file=f)
            elif query.tables[i].columns[j].type == "range":
                print("    unsigned int u_t%s_c%s, unsigned int l_t%s_c%s, __global unsigned int* t%s_c%s_ls, __global unsigned int* t%s_c%s_ss," % (i,j,i,j,i,j,i,j), file=f)
            else:
                raise Exception("Unknown column type.")

    print("    __global double* estimates, unsigned int skn) {", file=f)
    print("    for(unsigned int offset = 0; offset < skn; offset += get_global_size(0)){", file=f)
    print("        if(get_global_id(0) + offset < skn){", file=f)
    for i,col in enumerate(icols):
        for j in col:
            print("            unsigned int my_t%s_c%s_ls = t%s_c%s_ls[get_global_id(0)+offset]; int my_t%s_c%s_ss = is_set(t%s_c%s_ss[(get_global_id(0)+offset)/32],(get_global_id(0)+offset) %% 32);" % (i,j,i,j,i,j,i,j), file=f)   
    
    print("            estimates[get_global_id(0)+offset] = %s " % (" * ".join(["t%s_sketches[get_global_id(0)+offset]" % x for x in range(0,len(query.tables))])), end=' ', file=f)
    
    for i,col in enumerate(icols):
        for j in col:
            if query.tables[i].columns[j].type == "point":
                print("* ech3(t%s_c%s, my_t%s_c%s_ls, my_t%s_c%s_ss)" % (i,j,i,j,i,j), end=' ', file=f)
            elif query.tables[i].columns[j].type == "range":
                print("* range_ech3(u_t%s_c%s, l_t%s_c%s, my_t%s_c%s_ls, my_t%s_c%s_ss)" % (i,j,i,j,i,j,i,j), end=' ', file=f)
            else:
                raise Exception("Invalid column type.")
                
    print(";", file=f)
    print("        }", file=f)
    print("    }", file=f)
    print("}", file=f)
    
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
            
        print("""
int main( int argc, const char* argv[] ){
    parameters p;
    compute::device device = compute::system::default_device();
    p.ctx = compute::context(device);
    p.queue=compute::command_queue(p.ctx, device);
""", file=cf)
        print("""
    std::ifstream t("./%s_kernels.cl");
    t.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    std::string str((std::istreambuf_iterator<char>(t)),
    std::istreambuf_iterator<char>());
""" % i, file=cf)
        print("""
    compute::program pr = compute::program::create_with_source(str,p.ctx);
    try{
        std::ostringstream oss;
        pr.build(oss.str());
    } catch(const std::exception& ex){
        std::cout << pr.build_log() << std::endl;
    }
        """, file=cf)
        print("    p.iteration = atoi(argv[2]);", file=cf)
        print("    std::stringstream iteration_stream;", file=cf)
        print("    iteration_stream << \"./iteration\" << std::setw(2) << std::setfill('0') << argv[2];", file=cf)
        print("    p.skn = (unsigned int) atoll(argv[1]);", file=cf)
        print("    p.estimates = compute::vector<double>(p.skn, p.ctx);", file=cf)
        for j,t in enumerate(query.tables):
            print("    p.ts%s= %s;" % (j,ts[j]), file=cf)
            print("    p.sk_t%s = compute::vector<long>(p.skn, p.ctx);" % j, file=cf)
            for k,c in enumerate(t.columns):
                print("    std::stringstream t%s_c%s_stream ;" % (j,k), file=cf)
                print("    t%s_c%s_stream <<  \"./table_%s_%s.dump\";" % (j,k,t.tid,c.cid), file=cf) 
                print("    std::string t%s_c%s_string = t%s_c%s_stream.str();" % (j,k,j,k), file=cf)
                print("    unsigned int* t%s_c%s = readUArrayFromFile(t%s_c%s_string.c_str());" % (j,k,j,k), file=cf)
                print("    p.t%s_c%s = toGPUVector(t%s_c%s, p.ts%s, p.ctx, p.queue);" % (j,k,j,k,j), file=cf)

        print("    boost::random::mt19937 gen;", file=cf)
        
        for x, cols in enumerate(icols):
            for y in cols: 
                print("    unsigned int* t%s_c%s_lseed =  (unsigned int*) malloc(sizeof(unsigned int)*p.skn);" % (x,y), file=cf)
                print("    unsigned int* t%s_c%s_sseed =  (unsigned int*) malloc(((p.skn-1)/(sizeof(unsigned int)*8) +1)*sizeof(8));" % (x,y), file=cf)
                
                print("    for(unsigned int i = 0; i < p.skn;  i++ ){", file=cf)
                print("       t%s_c%s_lseed[i] = gen();" % (x,y), file=cf)
                print("    }", file=cf)
                
                print("    for(unsigned int i = 0; i < ((p.skn-1)/(sizeof(unsigned int)*8) +1);  i++ ){", file=cf) 
                print("       t%s_c%s_sseed[i] = gen();"  % (x,y), file=cf)
                print("    }", file=cf)

                print("    p.t%s_c%s_lseed =  toGPUVector(t%s_c%s_lseed, p.skn, p.ctx, p.queue);" % (x,y,x,y), file=cf)
                print("    p.t%s_c%s_sseed =  toGPUVector(t%s_c%s_sseed, ((p.skn-1)/(sizeof(unsigned int)*8) +1), p.ctx, p.queue);" % (x,y,x,y), file=cf)                
                
        
        for j,p1,p2 in jpairs:
            print("    unsigned int* j%s_t%s_c%s_t%s_c%s_lseed =  (unsigned int*) malloc(sizeof(unsigned int)*p.skn);" % (j,p1[0],p1[1],p2[0],p2[1]), file=cf)
            print("    unsigned int* j%s_t%s_c%s_t%s_c%s_sseed =  (unsigned int*) malloc(((p.skn-1)/(sizeof(unsigned int)*8) +1)*sizeof(unsigned int));" % (j,p1[0],p1[1],p2[0],p2[1]), file=cf)
            
            print("    for(unsigned int i = 0; i < p.skn;  i++ ){", file=cf)
            print("       j%s_t%s_c%s_t%s_c%s_lseed[i] = gen();" % (j,p1[0],p1[1],p2[0],p2[1]), file=cf)
            print("    }", file=cf)
            
            print("    for(unsigned int i = 0; i < ((p.skn-1)/(sizeof(unsigned int)*8) +1);  i++ ){", file=cf) 
            print("       j%s_t%s_c%s_t%s_c%s_sseed[i] = gen();"  % (j,p1[0],p1[1],p2[0],p2[1]), file=cf)
            print("    }", file=cf)
 
            print("    p.j%s_t%s_c%s_t%s_c%s_lseed = toGPUVector(j%s_t%s_c%s_t%s_c%s_lseed, p.skn, p.ctx, p.queue);" % (j,p1[0],p1[1],p2[0],p2[1],j,p1[0],p1[1],p2[0],p2[1]), file=cf)
            print("    p.j%s_t%s_c%s_t%s_c%s_sseed = toGPUVector(j%s_t%s_c%s_t%s_c%s_sseed, ((p.skn-1)/(sizeof(unsigned int)*8) +1), p.ctx, p.queue); " % (j,p1[0],p1[1],p2[0],p2[1],j,p1[0],p1[1],p2[0],p2[1]), file=cf)    
            
        for j,t in enumerate(query.tables):    
            print("    p.t%s_construct_sketch = pr.create_kernel(\"t%s_construct_sketch\");" % (j,j), file=cf) 
        print("    p.multiply_sketches = pr.create_kernel(\"multiply_sketches\");", file=cf)
        print("    sketch_contruction(&p);", file=cf)
        
        print("    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";", file=cf)
        print("    p.j_test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());", file=cf)
        
        for i,indices in enumerate(icols):
            for j in indices:
                if query.tables[i].columns[j].type == "point": 
                    print("    std::string j_p_t%s_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                    print("    p.j_p_t%s_c%s = readUArrayFromFile(j_p_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                elif query.tables[i].columns[j].type == "range": 
                    print("    std::string j_u_t%s_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                    print("    std::string j_l_t%s_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                    print("    p.j_u_t%s_c%s = readUArrayFromFile(j_u_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                    print("    p.j_l_t%s_c%s = readUArrayFromFile(j_l_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                else:
                    raise Exception("Unknown column type.")
        
        print(file=cf)
        print("    join_test(&p);", file=cf)
        print("}", file=cf)
