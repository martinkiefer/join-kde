 #Code generator for gauss kernel
import Utils
local_size = 64

def generatePreamble(f):
    print("""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef M_SQRT2
     #define M_SQRT2     1.41421356237309504880168872420969808
#endif
typedef double T;
""", file=f)

def rangeEstimateFunction(f):
    print("""
unsigned int range(unsigned int v, unsigned int u, unsigned int l){
    if(v == 0){
        return 0;
    }
    return v >= l && v <= u;
}
""", file=f)

def pointEstimateFunction(f):
    print("""
unsigned int point(unsigned int v, unsigned int p){
    return v == p;
}
""", file=f)

def generateEstimateKernel(f,kname,qtypes):
    print("__kernel void %s(" % kname, file=f)
    for i,k in enumerate(qtypes):
        if k == "range":
            print("    __global unsigned int* c%s,  unsigned int  u%s, unsigned int l%s, " % (i,i,i), file=f)
        elif k == "point":
            print("    __global unsigned int* c%s, unsigned int p%s, " % (i,i), file=f)
        else:
            raise Exception("Unsupported kernel.")
    print("    __global unsigned long* o, unsigned int ss){", file=f)
    print("        unsigned int counter = 0;", file=f)
    print("        for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){", file=f)
    print("            if (offset + get_global_id(0) < ss){", file=f)
    for i,k in enumerate(qtypes):
        if k == "point":
            print("                unsigned int ec%s = point(c%s[offset+get_global_id(0)], p%s);" % (i,i,i), file=f)
        elif k == "range":
            print("                unsigned int ec%s = range(c%s[offset+get_global_id(0)], u%s, l%s);" % (i,i,i,i), file=f)
        else:
            raise Exception("Unsupported kernel.")
    print("                counter += 1 ", end=' ', file=f)
    for i,k in enumerate(qtypes):
        print("&& ec%s" % i, end=' ', file=f)
    print(";", file=f)
    print("            }", file=f)
    print("        }", file=f)
    print("        o[get_global_id(0)] = counter;", file=f)
    print("}", file=f)
    print(file=f)
    
def generateCIncludes(f):
    print("""
    
#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>
#include <nlopt.h>
#include <sstream>
#include <cmath>

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>

namespace compute = boost::compute;
""", file=f)

def generateGPUJoinSampleCode(i,query,estimator,stats,cu_factor):
    ts, dv = stats
    qtype = []
    remap = []
    #Generate Kernels    
    with open("./%s_kernels.cl" % i,'w') as cf:
        generatePreamble(cf)
        
        cols = Utils.generateInvariantColumns(query)

        for j,indices in enumerate(cols):
            qtype.extend([query.tables[j].columns[index].type for index in indices ])
            remap.extend([(j,index) for index in indices ])
        rangeEstimateFunction(cf)
        pointEstimateFunction(cf)
        generateEstimateKernel(cf,"estimate",qtype)
    with open("./%s_GPUJS.cpp" % i,'w') as cf:
        generateCIncludes(cf)
        generateGPUJoinSampleParameterArray(cf,query,estimator,qtype)
        Utils.generateGPUVectorConverterFunction(cf)
        Utils.generateUintFileReaderFunction(cf)
        generateGPUJoinSampleEstimateFunction(cf,query,estimator,qtype)
        generateGPUJoinSampleTestWrapper(cf,query,estimator,qtype)
        
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

        #Read table sizes and read columns into memory and transfer to device the GPU
        print("    std::stringstream iteration_stream;", file=cf)
        print("    p.iteration = (unsigned int) atoi(argv[2]);", file=cf)
        print("    iteration_stream << \"./iteration\" << std::setw(2) << std::setfill('0') << argv[2];", file=cf)
        print("    p.ss = atoi(argv[1]);", file=cf)
        print("    p.local = 64;", file=cf)
        print("    p.global = std::min((size_t) p.ctx.get_device().compute_units()*%s, ((p.ss-1)/p.local+1)*p.local);" % cu_factor, file=cf)
        print("    p.local = std::min(p.local,p.global);", file=cf)
        print("    p.ts = %s;" % (ts), file=cf)
        for cid,kernel in enumerate(qtype):
            print("    std::stringstream s_c%s_stream ;" % (cid), file=cf)
            print("    s_c%s_stream << iteration_stream.str() << \"/jsample_\" << atoi(argv[1]) << \"_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf) 
            print("    std::string s_c%s_string = s_c%s_stream.str();" % (cid,cid), file=cf)
            print("    unsigned int* s_c%s = readUArrayFromFile(s_c%s_string.c_str());" % (cid,cid), file=cf)
            print("    p.s_c%s = toGPUVector(s_c%s, p.ss, p.ctx, p.queue);" % (cid,cid), file=cf)
            print(file=cf)   
        print("""
    compute::program pr = compute::program::create_with_source(str,p.ctx);
    try{
        std::ostringstream oss;
        pr.build(oss.str());
    } catch(const std::exception& ex){
        std::cout << pr.build_log() << std::endl;
    }
        """, file=cf)
        print("    p.out = compute::vector<unsigned long>(p.global, p.ctx);", file=cf)
        print("    p.estk = pr.create_kernel(\"estimate\");", file=cf)

        print("    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";", file=cf)
        print("    p.test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());", file=cf)

        for cid,ty in enumerate(qtype):
            if ty == "range":
                print("    std::string test_l_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                print("    p.test_l_c%s= readUArrayFromFile(test_l_c%s_string.c_str());" % (cid,cid), file=cf)
                print("    std::string test_u_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                print("    p.test_u_c%s = readUArrayFromFile(test_u_c%s_string.c_str());" % (cid,cid), file=cf)
            elif ty == "point":
                print("    std::string test_p_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                print("    p.test_p_c%s = readUArrayFromFile(test_p_c%s_string.c_str());" % (cid,cid), file=cf) 
            else:
                raise Exception("I don't know this ctype.")

        print(file=cf)
        print("    join_test(&p);", file=cf)
        print("}", file=cf) 
        
        
def generateGPUJoinSampleParameterArray(f,query,estimator,qtypes):
    print("""
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
""", file=f)
    print("    unsigned int iteration;", file=f)
    print("    size_t ss;", file=f)
    print("    size_t global;", file=f)
    print("    size_t local;", file=f)
    print("    unsigned int ts;", file=f)
    print("    compute::kernel estk;", file=f)
    for cid,kernel in enumerate(qtypes):
        print("    compute::vector<unsigned int> s_c%s;" % (cid), file=f)
            
    for cid,kernel in enumerate(qtypes):
        if kernel == "range":
            print("    unsigned int* test_l_c%s;" % (cid), file=f)
            print("    unsigned int* test_u_c%s;" % (cid), file=f)
        else:
            print("    unsigned int* test_p_c%s;" % (cid), file=f) 
            
    print("    compute::vector<unsigned long> out;", file=f)
    print("    unsigned int* test_cardinality;", file=f)

    print("""
} parameters;
""", file=f)


def generateGPUJoinSampleEstimateFunction(f, query, estimator, qtypes):
     print("double join_estimate_instance(parameters* p", file=f)
     for cid, qtype in enumerate(qtypes):
         # Start with computing the invariant contributions
         if qtype == "range":
             print("    , unsigned int u_c%s, unsigned int l_c%s" % (cid, cid), file=f)
         else:
             print("    , unsigned int p_c%s" % (cid), file=f)

     print("){", file=f)
     print("    p->estk.set_args(", end=' ', file=f)
     for cid, qtype in enumerate(qtypes):
         if qtype == "range":
             print("p->s_c%s, u_c%s, l_c%s, " % (cid, cid, cid), end=' ', file=f)
         else:
             print("p->s_c%s, p_c%s, " % (cid, cid), end=' ', file=f)
     print(" p->out, (unsigned int) p->ss", end=' ', file=f)
     print(");", file=f)
     print("    boost::compute::event ev = p->queue.enqueue_nd_range_kernel(p->estk,1,NULL,&(p->global), &(p->local));", file=f)
     # print >>f, "    ev.wait();"

     print("    unsigned long est = 0;", file=f)
     print("    boost::compute::reduce(p->out.begin(), p->out.begin()+std::min(p->global, p->ss), &est, p->queue);", file=f)
     print("    p->queue.finish();", file=f)
     print("    return est * ((double) p->ts)/p->ss;", file=f)
     # At this point, we need a
     print("}", file=f)


def generateGPUJoinSampleTestWrapper(f,query,estimator,qtypes):
    print("double join_test(parameters* p){", file=f)
    print("    double objective = 0.0;", file=f)
    print("    double est = 0.0;", file=f)
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.test, file=f)
    print("       auto begin = std::chrono::high_resolution_clock::now();", file=f)
    print("            est = join_estimate_instance(p", end=' ', file=f)

    for cid, qtype in enumerate(qtypes):
        if qtype == "range":
            print(", p->test_u_c%s[i]" % (cid), end=' ', file=f)
            print(", p->test_l_c%s[i]" % (cid), end=' ', file=f)
        else:
            print(", p->test_p_c%s[i]" % (cid), end=' ', file=f)
    print(");", file=f)
    print("        auto end = std::chrono::high_resolution_clock::now();", file=f)
    print("        double trues = p->test_cardinality[i];", file=f)
    print("        objective += (est-trues)*(est-trues);", file=f)
    Utils.printObjectiveLine(f,"p->ss")
    print("    }", file=f)
    print("    return objective/%s;" % estimator.test, file=f)
    print("}", file=f)
