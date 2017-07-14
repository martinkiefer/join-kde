 #Code generator for gauss kernel
import Utils
local_size = 64
from KDECodeGenerator import generatePreamble
from KDECodeGenerator import generateGPUJKDETestWrapper
from JoinGraph import constructJoinGraph
import operator
        
def prod(iterable):
    return reduce(operator.mul, iterable, 1)        
        
def generateBinarySearchCode(f):
    print >>f, """
unsigned int binarySearch(__global unsigned int* x, unsigned int val,unsigned int len){
    unsigned int l = 0;
    unsigned int u = len-1;

    if(x[l] >= val) return 0;
    if(x[u] < val) return len;

    while(u - l != 1){
        int c = l + (u-l)/2;
        if(x[c] >= val)
            u=c;
        else
            l=c;

    }
    return u;
}    
    """

def generateJoinEstimateKernel(f,query,estimator,stats):
    print >>f, "__kernel void estimate("
    icols = Utils.generateInvariantColumns(query)   
    jcols = Utils.generateJoinColumns(query)

    graph = constructJoinGraph(query)
    tids = graph.collectTableIDs()
    pairs = graph.collectJoinPairs()

    _ , dvals = stats


    for x,t in enumerate(tids):
        for jc in jcols[t]:
            print >>f, "    __global unsigned int* t%s_c%s," % (t,jc),
        if x > 0:
            print >>f, "    unsigned int n_t%s," % (t)
    print >>f, "    __global unsigned long *contributions, unsigned int ss){"

    print >> f
    #We start of with table 1.
    print >> f, "    unsigned long sum = 0;"
    print >>f, "     for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
    print >>f, "        if (offset + get_global_id(0) < ss){"

    graph.generateJoinEstimateKernelBottomUp(f, query, estimator)
    print >> f, "    sum++;"
    graph.generateJoinEstimateKernelTopDown(f, query)

    print >>f, "        }"
    print >>f, "    }"
    print >>f, "    if (get_global_id(0) < ss) contributions[get_global_id(0)] = sum;"
    print >>f, "}"

#Classes representing a left-deep join tree
    
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
#include <boost/compute/algorithm/sort_by_key.hpp>
#include <boost/compute/algorithm/gather.hpp>
#include <boost/compute/algorithm/inclusive_scan.hpp>
#include <boost/compute/algorithm/exclusive_scan.hpp>
#include <boost/compute/algorithm/scatter.hpp>
#include <boost/compute/algorithm/scatter_if.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/functional/math.hpp>
#include <boost/compute/iterator/counting_iterator.hpp>

namespace compute = boost::compute;
"""

def generateGPUSampleCode(i,query,estimator,stats,cu_factor):
    ts, dv = stats
    graph = constructJoinGraph(query)
    tids = graph.collectTableIDs()

    #Generate Kernels    
    with open("./%s_kernels.cl" % i,'w') as cf:
        generatePreamble(cf)
        
        print >>cf, "//"
        graph.generateTableEstimateKernel(cf, query, estimator, stats)
        generateBinarySearchCode(cf)
        generateJoinEstimateKernel(cf,query,estimator,stats)
        print >>cf, "//"

        
    with open("./%s_GPUS.cpp" % i,'w') as cf:
        generateCIncludes(cf)
        generateGPUSampleParameterArray(cf,query,estimator)
        Utils.generateGPUVectorConverterFunction(cf)
        Utils.generateUintFileReaderFunction(cf)
        Utils.generateScottBWFunction(cf)
        generateGPUSampleEstimateFunction(cf,graph,query,estimator,prod(ts.values())**-1.0,stats,cu_factor)
        #There is no reason why we shouldn't use the estimate function from GPUJKDE.
        generateGPUJKDETestWrapper(cf,query,estimator)
        
        cols = Utils.generateInvariantColumns(query)
        jcols = Utils.generateJoinColumns(query)
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
        print >>cf, "    p.iteration = (unsigned int) atoi(argv[%s]);" % (len(query.tables)+1)
        print >>cf, "    iteration_stream << \"./iteration\" << std::setw(2) << std::setfill('0') << argv[%s];" % (len(query.tables)+1)
        for j,t in enumerate(query.tables):
            print >>cf, "    p.ss%s = atoi(argv[%s]);" % (j,j+1)
            print >>cf, "    p.ts%s= %s;" % (j,ts[j])
            for k,c in enumerate(t.columns):
                print >>cf, "    std::stringstream s_t%s_c%s_stream ;" % (j,k)
                print >>cf, "    s_t%s_c%s_stream << iteration_stream.str() << \"/sample_\" << atoi(argv[%s]) << \"_%s_%s.dump\";" % (j,k,j+1,t.tid,c.cid) 
                print >>cf, "    std::string s_t%s_c%s_string = s_t%s_c%s_stream.str();" % (j,k,j,k)
                print >>cf, "    unsigned int* s_t%s_c%s = readUArrayFromFile(s_t%s_c%s_string.c_str());" % (j,k,j,k)
                print >>cf, "    p.s_t%s_c%s = toGPUVector(s_t%s_c%s, p.ss%s, p.ctx, p.queue);" % (j,k,j,k,j)
            print >>cf

        for t,cs in enumerate(jcols):
            if cols[t]:
                for c in cs:
                    print >> cf, "    p.sr_t%s_c%s = compute::vector<unsigned int>(p.ss%s, p.ctx);" % (t,c,t)
        print >> cf, "    p.final_contributions = compute::vector<unsigned long>(p.ss%s, p.ctx);" % tids[0]
        print >>cf, """
    compute::program pr = compute::program::create_with_source(str,p.ctx);
    try{
        std::ostringstream oss;
        pr.build(oss.str());
    } catch(const std::exception& ex){
        std::cout << pr.build_log() << std::endl;
    }
        """
        for j,t in enumerate(query.tables):
            if len(cols[j]) > 0:
                print >>cf, "    p.invk%s = pr.create_kernel(\"invk_t%s\");" % (j,j)   
                print >>cf, "    p.inv_t%s = compute::vector<double>(p.ss%s, p.ctx);" % (j,j)
                print >> cf, "   p.invr_t%s = compute::vector<double>(p.ss%s, p.ctx);" % (j, j)
        print >>cf, "    p.estimate = pr.create_kernel(\"estimate\");"
        print >>cf

        for t, tab in enumerate(query.tables):
            print >> cf, "    p.map_t%s = compute::vector<unsigned int >(p.ss%s+1, p.ctx);" % (t,t)
            print >> cf, "    p.count_t%s = compute::vector<int >(p.ss%s+1, p.ctx);" % (t,t)
            print >> cf, "    p.count_t%s[0] = -1;" % t
            
        print >>cf, "    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";"
        print >>cf, "    p.j_test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());"
        
        for i,indices in enumerate(cols):
            if len(indices) != 0:
                for j in indices:
                    if query.tables[i].columns[j].type == "range":
                        print >>cf, "    std::string j_l_t%s_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                        print >>cf, "    p.j_l_t%s_c%s= readUArrayFromFile(j_l_t%s_c%s_string.c_str());" % (i,j,i,j)
                        print >>cf, "    std::string j_u_t%s_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                        print >>cf, "    p.j_u_t%s_c%s = readUArrayFromFile(j_u_t%s_c%s_string.c_str());" % (i,j,i,j)
                    elif query.tables[i].columns[j].type == "point":
                        print >>cf, "    std::string j_p_t%s_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                        print >>cf, "    p.j_p_t%s_c%s = readUArrayFromFile(j_p_t%s_c%s_string.c_str());" % (i,j,i,j)
                    else:
                        raise Exception("Unsupported ctype.")
        
        print >>cf
        print >>cf, "    join_test(&p);"
        print >>cf, "}"   

        
#Generate parameter struct that is passed to the estimation/gradient functions
def generateGPUSampleParameterArray(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    jcols = Utils.generateJoinColumns(query)
    print >>f, """
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
"""
    print >>f, "    unsigned int iteration;"
    for j,t in enumerate(query.tables):
        print >>f, "    size_t ss%s;" % (j) 
        print >>f, "    unsigned int ts%s;" % (j)
        if len(cols[j]) > 0:
            print >>f, "    compute::kernel invk%s;" % (j)
            print >>f, "    compute::vector<double> inv_t%s;" % (j)
            print >> f, "    compute::vector<double> invr_t%s;" % (j)
        for k,c in enumerate(t.columns):
            print >>f, "    compute::vector<unsigned int> s_t%s_c%s;" % (j,k)
            print >>f, "    double bw_t%s_c%s;" % (j,k)
        print >>f
    print >>f, "    compute::kernel estimate;"

    for t,tab in enumerate(query.tables):
        print >>f, "    compute::vector<unsigned int> map_t%s;" % t
        print >>f, "    compute::vector<int> count_t%s;" % t

    for t,_ in enumerate(jcols):
        for c in jcols[t]:
            print >> f, "    compute::vector<unsigned int> sr_t%s_c%s;" % (t, c)
    print >>f, "    compute::vector<unsigned long> final_contributions;"
    #Training

    print >>f    
    print >>f, "    unsigned int* j_test_cardinality;"
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        if len(indices) != 0:
            for j in indices:
                if query.tables[i].columns[j].type == "range":
                    print >>f, "    unsigned int* j_l_t%s_c%s;" % (i,j)
                    print >>f, "    unsigned int* j_u_t%s_c%s;" % (i,j)
                elif query.tables[i].columns[j].type == "point":
                    print >>f, "    unsigned int* j_p_t%s_c%s;" % (i,j)
                else:
                    raise Exception("Unknown ctype.")
                
    print >>f, """
} parameters;
"""


def generateGPUSampleEstimateFunction(f, nodes, query, estimator, limit, stats, cu_factor):
     icols = Utils.generateInvariantColumns(query)
     jcols = Utils.generateJoinColumns(query)
     ts, dv = stats

     print >> f, "double join_estimate_instance(parameters* p"
     for i, indices in enumerate(icols):
         # Start with computing the invariant contributions
         if len(indices) != 0:
             for j in indices:
                 if query.tables[i].columns[j].type == "range":
                     print >> f, "    , unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (i, j, i, j)
                 elif query.tables[i].columns[j].type == "point":
                     print >> f, "    , unsigned int p_t%s_c%s" % (i, j)
                 else:
                     raise Exception("Unknown ctype.")
     print >> f
     print >> f, "){"

     nodes.generateTableCode(f, query, estimator, limit, cu_factor)

     # Next, generate the limits
     pairs = nodes.collectJoinPairs()
     tids = nodes.collectTableIDs()


     print >> f, "    size_t local = 64;"
     print >> f, "    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((rss_t%s-1)/local+1)*local);" % (cu_factor,tids[0])
     print >> f, "    p->estimate.set_args(",

     for x, t in enumerate(tids):
         for jc in jcols[t]:
             if icols[t]:
                 print >> f, "    p->sr_t%s_c%s," % (t, jc),
             else:
                 print >> f, "    p->s_t%s_c%s," % (t, jc),
         if x > 0:
             print >> f, "    (unsigned int) rss_t%s," % (t)
     print >> f, "    p->final_contributions, (unsigned int) rss_t%s);" % tids[0]
     print >> f, "    p->queue.enqueue_nd_range_kernel(p->estimate,1,NULL,&(global), &(local));" 

     print >> f, "    unsigned long counter = 0.0;"
     print >> f, "    boost::compute::reduce(p->final_contributions.begin(), p->final_contributions.begin()+std::min(rss_t%s,global), &counter, p->queue);" % (
     tids[0])
     print >> f, "    p->queue.finish();"
     print >> f, "    double est = counter;"
     for i, _ in enumerate(query.tables):
         print >> f, "    est *= ((double) p->ts%s)/p->ss%s;" % (i, i)
     print >> f, "    return est;"
     print >> f, "}"
