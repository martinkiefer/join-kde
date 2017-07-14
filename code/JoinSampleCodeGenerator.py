 #Code generator for gauss kernel
import Utils
local_size = 64

def generatePreamble(f):
    print >>f, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef M_SQRT2
     #define M_SQRT2     1.41421356237309504880168872420969808
#endif
typedef double T;
"""

def rangeEstimateFunction(f):
    print >>f, """
unsigned int range(unsigned int v, unsigned int u, unsigned int l){
    if(v == 0){
        return 0;
    }
    return v >= l && v <= u;
}
"""

def pointEstimateFunction(f):
    print >>f, """
unsigned int point(unsigned int v, unsigned int p){
    return v == p;
}
"""

def generateEstimateKernel(f,kname,qtypes):
    print >>f, "__kernel void %s(" % kname
    for i,k in enumerate(qtypes):
        if k == "range":
            print >>f, "    __global unsigned int* c%s,  unsigned int  u%s, unsigned int l%s, " % (i,i,i)
        elif k == "point":
            print >>f, "    __global unsigned int* c%s, unsigned int p%s, " % (i,i)
        else:
            raise Exception("Unsupported kernel.")
    print >>f, "    __global unsigned long* o, unsigned int ss){"
    print >>f, "        unsigned int counter = 0;"
    print >>f, "        for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
    print >>f, "            if (offset + get_global_id(0) < ss){"
    for i,k in enumerate(qtypes):
        if k == "point":
            print >>f, "                unsigned int ec%s = point(c%s[offset+get_global_id(0)], p%s);" % (i,i,i)
        elif k == "range":
            print >>f, "                unsigned int ec%s = range(c%s[offset+get_global_id(0)], u%s, l%s);" % (i,i,i,i)
        else:
            raise Exception("Unsupported kernel.")
    print >>f, "                counter += 1 ",
    for i,k in enumerate(qtypes):
        print >>f, "&& ec%s" % i,
    print >>f, ";"
    print >>f, "            }"
    print >>f, "        }"
    print >>f, "        o[get_global_id(0)] = counter;"
    print >>f, "}"
    print >>f
    
def generateCIncludes(f):
    print >>f, """
    
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
"""

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

        #Read table sizes and read columns into memory and transfer to device the GPU
        print >>cf, "    std::stringstream iteration_stream;"
        print >>cf, "    p.iteration = (unsigned int) atoi(argv[2]);"
        print >>cf, "    iteration_stream << \"./iteration\" << std::setw(2) << std::setfill('0') << argv[2];"
        print >>cf, "    p.ss = atoi(argv[1]);"
        print >> cf, "    p.local = 64;"
        print >> cf, "    p.global = std::min((size_t) p.ctx.get_device().compute_units()*%s, ((p.ss-1)/p.local+1)*p.local);" % cu_factor
        print >>cf, "    p.ts = %s;" % (ts)
        for cid,kernel in enumerate(qtype):
            print >>cf, "    std::stringstream s_c%s_stream ;" % (cid)
            print >>cf, "    s_c%s_stream << iteration_stream.str() << \"/jsample_\" << atoi(argv[1]) << \"_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid) 
            print >>cf, "    std::string s_c%s_string = s_c%s_stream.str();" % (cid,cid)
            print >>cf, "    unsigned int* s_c%s = readUArrayFromFile(s_c%s_string.c_str());" % (cid,cid)
            print >>cf, "    p.s_c%s = toGPUVector(s_c%s, p.ss, p.ctx, p.queue);" % (cid,cid)
            print >>cf   
        print >>cf, """
    compute::program pr = compute::program::create_with_source(str,p.ctx);
    try{
        std::ostringstream oss;
        pr.build(oss.str());
    } catch(const std::exception& ex){
        std::cout << pr.build_log() << std::endl;
    }
        """
        print >>cf, "    p.out = compute::vector<unsigned long>(p.global, p.ctx);"
        print >>cf, "    p.estk = pr.create_kernel(\"estimate\");"

        print >>cf, "    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";"
        print >>cf, "    p.test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());"

        for cid,ty in enumerate(qtype):
            if ty == "range":
                print >>cf, "    std::string test_l_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                print >>cf, "    p.test_l_c%s= readUArrayFromFile(test_l_c%s_string.c_str());" % (cid,cid)
                print >>cf, "    std::string test_u_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                print >>cf, "    p.test_u_c%s = readUArrayFromFile(test_u_c%s_string.c_str());" % (cid,cid)
            elif ty == "point":
                print >>cf, "    std::string test_p_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                print >>cf, "    p.test_p_c%s = readUArrayFromFile(test_p_c%s_string.c_str());" % (cid,cid) 
            else:
                raise Exception("I don't know this ctype.")

        print >>cf
        print >>cf, "    join_test(&p);"
        print >>cf, "}" 
        
        
def generateGPUJoinSampleParameterArray(f,query,estimator,qtypes):
    print >>f, """
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
"""
    print >>f, "    unsigned int iteration;"
    print >>f, "    size_t ss;"
    print >>f, "    size_t global;"
    print >>f, "    size_t local;"
    print >>f, "    unsigned int ts;"
    print >>f, "    compute::kernel estk;"
    for cid,kernel in enumerate(qtypes):
        print >>f, "    compute::vector<unsigned int> s_c%s;" % (cid)
            
    for cid,kernel in enumerate(qtypes):
        if kernel == "range":
            print >>f, "    unsigned int* test_l_c%s;" % (cid)
            print >>f, "    unsigned int* test_u_c%s;" % (cid)
        else:
            print >>f, "    unsigned int* test_p_c%s;" % (cid) 
            
    print >>f, "    compute::vector<unsigned long> out;"
    print >>f, "    unsigned int* test_cardinality;"

    print >>f, """
} parameters;
"""


def generateGPUJoinSampleEstimateFunction(f, query, estimator, qtypes):
     print >> f, "double join_estimate_instance(parameters* p"
     for cid, qtype in enumerate(qtypes):
         # Start with computing the invariant contributions
         if qtype == "range":
             print >> f, "    , unsigned int u_c%s, unsigned int l_c%s" % (cid, cid)
         else:
             print >> f, "    , unsigned int p_c%s" % (cid)

     print >> f, "){"
     print >> f, "    p->estk.set_args(",
     for cid, qtype in enumerate(qtypes):
         if qtype == "range":
             print >> f, "p->s_c%s, u_c%s, l_c%s, " % (cid, cid, cid),
         else:
             print >> f, "p->s_c%s, p_c%s, " % (cid, cid),
     print >> f, " p->out, (unsigned int) p->ss",
     print >> f, ");"
     print >> f, "    boost::compute::event ev = p->queue.enqueue_nd_range_kernel(p->estk,1,NULL,&(p->global), &(p->local));"
     # print >>f, "    ev.wait();"

     print >> f, "    unsigned long est = 0;"
     print >> f, "    boost::compute::reduce(p->out.begin(), p->out.begin()+std::min(p->global, p->ss), &est, p->queue);"
     print >> f, "    p->queue.finish();"
     print >> f, "    return est * ((double) p->ts)/p->ss;"
     # At this point, we need a
     print >> f, "}"


def generateGPUJoinSampleTestWrapper(f,query,estimator,qtypes):
    print >>f, "double join_test(parameters* p){"
    print >>f, "    double objective = 0.0;"
    print >>f, "    double est = 0.0;"
    print >>f, "    int first = 1;"

    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.test
    print >> f, "       auto begin = std::chrono::high_resolution_clock::now();"
    print >>f, "        if(first ",

    for cid, qtype in enumerate(qtypes):
        if qtype == "range":
            print >>f, "|| p->test_l_c%s[i] != p->test_l_c%s[i-1] " % (cid,cid),
            print >>f, "|| p->test_u_c%s[i] != p->test_u_c%s[i-1] " % (cid,cid),
        else:
            print >>f, "|| p->test_p_c%s[i] != p->test_p_c%s[i-1] " % (cid,cid),
    print >>f, "){"
    if hasattr(estimator, 'look_behind'):
        if estimator.look_behind:
            print >> f, "            first = 0;"
    else:
        print >>f, "            first = 0;"
    print >>f, "            est = join_estimate_instance(p",

    for cid, qtype in enumerate(qtypes):
        if qtype == "range":
            print >>f, ", p->test_u_c%s[i]" % (cid),
            print >>f, ", p->test_l_c%s[i]" % (cid),
        else:
            print >>f, ", p->test_p_c%s[i]" % (cid),
    print >>f, ");"
    print >>f, "        }"
    print >>f, "        auto end = std::chrono::high_resolution_clock::now();"
    print >>f, "        double trues = p->test_cardinality[i];"
    print >>f, "        objective += (est-trues)*(est-trues);"
    Utils.printObjectiveLine(f,"p->ss")
    print >>f, "    }"
    print >>f, "    return objective/%s;" % estimator.test
    print >>f, "}"
