 #Code generator for gauss kernel
import Utils
local_size = 64

class GaussKernel:
    def rangeEstimateFunction(self,f):
        print >>f, """
T gaussRangeEst(unsigned int v, T h, unsigned int u, unsigned int l){
    if(v == 0){
        return 0.0;
    }
    T up = ((T)u)-v;
    T lo = ((T)l)-v;
    return (erf(up/(M_SQRT2*h)) - erf(lo/(M_SQRT2*h)))*0.5;
}
"""

    def rangeGradientFunction(self, f ):
        print >>f, """
T gaussRangeGrad(unsigned int v, T h, unsigned int u, unsigned int l){
    if(v == 0){
        return 0.0;
    }
    T up = ((T)u)-v;
    T lo = ((T)l)-v;
    return (lo * exp((T)-1.0 * lo * lo / (2*h*h)) - up * exp((T)-1.0 * up * up / (2*h*h)))*0.5*0.5*M_SQRT2*M_2_SQRTPI/(h*h);
}
"""

    def rangeGradientConstant(self, f ,bwvar):
        return "M_2_SQRTPI*0.5*M_SQRT2 / (%s*%s)" % (bwvar,bwvar)

    def pointEstimateFunction(self, f ):
        print >>f, """
T gaussPointEst(unsigned int v, T h, unsigned int p){
    if(v == 0){
        return v==p;
    }
    T up = ((T) p)+0.5-v;
    T lo = ((T) p)-0.5-v;
    return (erf(up/(M_SQRT2*h)) - erf(lo/(M_SQRT2*h)))*0.5;
}
"""

    def pointGradientFunction(self, f ):
        print >>f, """
T gaussPointGrad(unsigned int v, T h, unsigned int p){
    if(v == 0){
        return 0.0;
    }
    T up = ((T) p)+0.5-v;
    T lo = ((T) p)-0.5-v;
    return (lo * exp((T)-1.0 * lo * lo / (2*h*h)) - up * exp((T)-1.0 * up * up / (2*h*h)))*0.5*0.5*M_SQRT2*M_2_SQRTPI/(h*h);
}
"""


#Code generator for categorical kernel
class CategoricalKernel:
    def pointEstimateFunction(self, f ):
        print >>f, """
T catPointEst(unsigned int v, T h, unsigned int p, unsigned int dvals){
    return (v == p) ? (1.0-h) : h/(dvals-1);
}
"""

    def pointEstimateConstant(self, f ,bw=None):
        return None

    def pointGradientFunction(self, f ):
        print >>f, """
T catPointGrad(unsigned int v, T h, unsigned int p, unsigned int dvals){
    return (v == p) ? -1.0 : 1.0/(dvals-1);
}
"""

    def pointGradientConstant(self, f ,bw=None):
        return None
        
def generateTableGradientContributionKernel(f,kname,kernels,dvals = None):
    print >>f, "__kernel void %s(" % kname
    for i,k in enumerate(kernels):
        if k == "GaussRange":
            print >>f, "    __global unsigned int* c%s, T h%s, unsigned int  u%s, unsigned int l%s, __global T* g%s," % (i,i,i,i,i)
        elif k == "GaussPoint" or k == "CategoricalPoint":
            print >>f, "    __global unsigned int* c%s, T h%s, unsigned int p%s, __global T* g%s," % (i,i,i,i)
        else:
            raise Exception("Unsupported kernel.")
    print >>f, "    __global T* o, unsigned int ss){"
    print >>f, "        T cont = 0.0;"
    for i,k in enumerate(kernels):
        print >>f, "        T gcont_c%s = 0.0;" % i

    print >>f, "        for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
    print >>f, "            if (offset + get_global_id(0) < ss){"
    for i,k in enumerate(kernels):
        if k == "GaussPoint":
            print >>f, "                T ec%s = gaussPointEst(c%s[offset+get_global_id(0)], h%s, p%s);" % (i,i,i,i)
            print >>f, "                T gc%s = gaussPointGrad(c%s[offset+get_global_id(0)], h%s, p%s);" % (i,i,i,i)
        elif k == "GaussRange":
            print >>f, "                T ec%s = gaussRangeEst(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (i,i,i,i,i)
            print >>f, "                T gc%s = gaussRangeGrad(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (i,i,i,i,i)
        elif k == "CategoricalPoint":
            print >>f, "                T ec%s = catPointEst(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (i,i,i,i,dvals[i])
            print >>f, "                T gc%s = catPointGrad(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (i,i,i,i,dvals[i])            
        else:
            raise Exception("Unsupported kernel.")
    print >>f, "                cont += 1.0 ",
    for i,k in enumerate(kernels):
        print >>f, "* ec%s" % i,
    print >>f, ";" 
    for i,_ in enumerate(kernels):
        print >>f, "                gcont_c%s += gc%s" % (i,i),
        for j,_ in enumerate(kernels):
            if i == j:
                continue  
            else:
                print >>f, "* ec%s" % j,  
        print >>f, ";"
    print >>f, "            }"
    print >>f, "        }"
    print >>f, "        if (get_global_id(0) < ss) o[get_global_id(0)]= cont;"
    for i,_ in enumerate(kernels):
        print >>f, "        if (get_global_id(0) < ss) g%s[get_global_id(0)] = gcont_c%s;" % (i,i)
    print >>f, "}"
    print >>f
    
def generateTableEstimateContributionKernel(f,kname,kernels,dvals = None):
    print >>f, "__kernel void %s(" % kname
    for i,k in enumerate(kernels):
        if k == "GaussRange":
            print >>f, "    __global unsigned int* c%s, T h%s, unsigned int  u%s, unsigned int l%s," % (i,i,i,i)
        elif k == "GaussPoint" or k == "CategoricalPoint":
            print >>f, "    __global unsigned int* c%s, T h%s, unsigned int p%s," % (i,i,i)
        else:
            raise Exception("Unsupported kernel.")
    print >>f, "    __global T* o, unsigned int ss){"
    print >>f, "        T cont = 0.0;"
    print >>f, "        for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
    print >>f, "            if (offset + get_global_id(0) < ss){"
    for i,k in enumerate(kernels):
        if k == "GaussPoint":
            print >>f, "                T ec%s = gaussPointEst(c%s[offset+get_global_id(0)], h%s, p%s);" % (i,i,i,i)
        elif k == "GaussRange":
            print >>f, "                T ec%s = gaussRangeEst(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (i,i,i,i,i)
        elif k == "CategoricalPoint":
            print >>f, "                T ec%s = catPointEst(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (i,i,i,i,dvals[i])
        else:
            raise Exception("Unsupported kernel.")
    print >>f, "                cont += 1.0 ",
    for i,k in enumerate(kernels):
        print >>f, "* ec%s" % i,
    print >>f, ";"
    print >>f, "            }"
    print >>f, "        }"
    print >>f, "        if (get_global_id(0) < ss) o[get_global_id(0)] = cont;"
    print >>f, "}"
    print >>f


def generatePreamble(f):
    print >>f, """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef M_SQRT2
     #define M_SQRT2     1.41421356237309504880168872420969808
#endif
typedef double T;
"""
    
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
def generateGPUJKDELocalTraining(cf,query,estimator, cu_factor):
            raise Exception("This code needs to be checked and should not be used right now.")
            for tid,t in enumerate(query.tables):
                #print >>cf, "    p.estk%s = pr.create_kernel(\"est_t%s\");" % (tid,tid)   
                print >> cf, "    p.local_t%s = 64;" % tid
                print >> cf, "    p.global_t%s = std::min((size_t) p.ctx.get_device().compute_units()*%s , ((p.ss%s-1)/p.local_t%s+1)*p.local_t%s);" % (tid,cu_factortid,tid,tid)
                print >>cf, "    p.gradk%s = pr.create_kernel(\"grad_t%s\");" % (tid,tid)   
                print >>cf, "    p.est_t%s = compute::vector<double>(p.global_t%s, p.ctx);" % (tid,tid) 
                print >>cf, "    std::string true_t%s_string = iteration_stream.str() + \"/training_table_%s_true.dump\";" % (tid,query.tables[tid].tid)
                print >>cf, "    p.true_t%s = readUArrayFromFile(true_t%s_string.c_str());" % (tid,tid)
                print >>cf
                for cid,col in enumerate(t.columns):
                    print >>cf, "    p.grad_t%s_c%s = compute::vector<double>(p.global_t%s, p.ctx);" % (tid,cid,tid) 
                    if estimator.kernels[tid][cid] == "GaussRange":
                        print >>cf, "    std::string l_t%s_c%s_string = iteration_stream.str() + \"/training_table_l_%s_%s.dump\";" % (tid,cid,query.tables[tid].tid,query.tables[tid].columns[cid].cid)
                        print >>cf, "    p.l_t%s_c%s= readUArrayFromFile(l_t%s_c%s_string.c_str());" % (tid,cid)
                        print >>cf, "    std::string u_t%s_c%s_string = iteration_stream.str() + \"/training_table_u_%s_%s.dump\";" % (tid,cid,query.tables[tid].tid,query.tables[tid].columns[cid].cid)
                        print >>cf, "    p.u_t%s_c%s = readUArrayFromFile(u_t%s_c%s_string.c_str());" % (tid,cid)
                    else:
                        print >>cf, "    std::string p_t%s_c%s_string = iteration_stream.str() + \"/training_table_p_%s_%s.dump\";" % (tid,cid,query.tables[tid].tid,query.tables[tid].columns[cid].cid)
                        print >>cf, "    p.p_t%s_c%s = readUArrayFromFile(p_t%s_c%s_string.c_str());" % (tid,cid,tid,cid) 
                    print >>cf
                    
                #Prepare dat bounds
                print >>cf, "    double ub%s[%s] = {0.0};" % (tid,len(t.columns)) 
                print >>cf, "    double lb%s[%s] = {0.0};" % (tid,len(t.columns)) 
                
                print >>cf, "    double minf%s = 0.0;" % tid
                print >>cf, "    double bw%s[%s] = {0.0};" % (tid,len(t.columns))
                for cid,col in enumerate(t.columns):
                    print >>cf, "    bw%s[%s] = p.bw_t%s_c%s;" % (tid,cid,tid,cid)
                    
                for cid,col in enumerate(t.columns):
                    if estimator.kernels[tid][cid] == "GaussRange" or estimator.kernels[tid][cid] == "GaussPoint":
                        print >>cf, "    ub%s[%s] = p.bw_t%s_c%s;" % (tid,cid, tid,cid) 
                        print >>cf, "    lb%s[%s] = 0.1;" % (tid,cid) 
                    elif estimator.kernels[tid][cid] == "CategoricalPoint":
                        print >>cf, "    lb%s[%s] = DBL_EPSILON;" % (tid,cid) 
                        print >>cf, "    ub%s[%s] = 1.0-DBL_EPSILON;" % (tid,cid)
                        print >>cf, "    bw%s[%s] = 1.0/p.ss%s;" % (tid,cid,tid) 
                #Prepare dat optimizer
#                print >>cf, """ 
#    nlopt_opt gopt%s = nlopt_create(NLOPT_GD_MLSL,%s);
#    nlopt_set_lower_bounds(gopt%s,lb%s);
#    nlopt_set_upper_bounds(gopt%s,ub%s);
#    nlopt_set_min_objective(gopt%s,obj_grad_t%s,&p);
#""" % (tid,len(t.columns),tid,tid,tid,tid,tid,tid)
#                print >>cf
#                print >>cf, """ 
#    nlopt_set_maxeval(gopt%s, %s);
#    nlopt_set_ftol_rel(gopt%s, %s);
#    nlopt_opt lopt%s = nlopt_create(NLOPT_LD_MMA,%s);
#    nlopt_set_lower_bounds(lopt%s,lb%s);
#    nlopt_set_upper_bounds(lopt%s,ub%s);
#    nlopt_set_local_optimizer(lopt%s, lopt%s);
#    int grc%s = nlopt_optimize(gopt%s, bw%s, &minf%s);
#    assert(grc%s >=0);
#""" % (tid,40,tid,"1e-10",tid,len(t.columns),tid,tid,tid,tid,tid,tid,tid,tid,tid,tid,tid)
                 
                print >>cf, """
    nlopt_opt opt%s = nlopt_create(NLOPT_LD_MMA,%s);
    nlopt_set_lower_bounds(opt%s,lb%s);
    nlopt_set_upper_bounds(opt%s,ub%s);
    nlopt_set_maxeval(opt%s, %s);
    nlopt_set_ftol_rel(opt%s, %s);
    nlopt_set_min_objective(opt%s,obj_grad_t%s,&p);
    int frc%s = nlopt_optimize(opt%s, bw%s, &minf%s);
    assert(frc%s >=0);    
""" % (tid,len(t.columns),tid,tid,tid,tid,tid,1000,tid,"1e-5",tid,tid,tid,tid,tid,tid,tid)
                for cid,col in enumerate(t.columns):
                    print >>cf, "    p.bw_t%s_c%s = bw%s[%s];" % (tid,cid,tid,cid)
                print >>cf
                    
def generateGPUKDECode(i,query,estimator,stats,cu_factor):
    ts, dv = stats
    kernels = []
    remap = []
    #Generate Kernels    
    with open("./%s_kernels.cl" % i,'w') as cf:
        generatePreamble(cf)
        
        gk = GaussKernel()
        gk.pointEstimateFunction(cf)
        gk.pointGradientFunction(cf)
        gk.rangeEstimateFunction(cf)
        gk.rangeGradientFunction(cf)
        
        
        ck = CategoricalKernel()
        ck.pointEstimateFunction(cf)
        ck.pointGradientFunction(cf)

        cols = Utils.generateInvariantColumns(query)
        for j,indices in enumerate(cols):
            kernels.extend([estimator.kernels[j][index] for index in indices ])
            remap.extend([(j,index) for index in indices ])
        generateTableGradientContributionKernel(cf,"grad",kernels,dv)
        generateTableEstimateContributionKernel(cf, "est", kernels, dv)
    with open("./%s_GPUKDE.cpp" % i,'w') as cf:
        generateCIncludes(cf)
        generateGPUKDEParameterArray(cf,query,estimator,kernels)
        Utils.generateGPUVectorConverterFunction(cf)
        Utils.generateUintFileReaderFunction(cf)
        Utils.generateDoubleFileReaderFunction(cf)
        Utils.generateDoubleDumper(cf)
        Utils.generateFileCheckFunction(cf)
        Utils.generateScottBWFunction(cf)
        generateGPUKDEEstimateFunction(cf,query,estimator,kernels)
        generateGPUKDETestWrapper(cf,query,estimator,kernels)
        generateGPUKDEEstGrad(cf,query,estimator,kernels)
        generateGPUKDEObjective(cf,query,estimator,kernels)
        
        
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
        print >>cf, "    p.ts = %s;" % (ts)
        for cid,kernel in enumerate(kernels):
            print >>cf, "    std::stringstream s_c%s_stream ;" % (cid)
            print >>cf, "    s_c%s_stream << iteration_stream.str() << \"/jsample_\" << atoi(argv[1]) << \"_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid) 
            print >>cf, "    std::string s_c%s_string = s_c%s_stream.str();" % (cid,cid)
            print >>cf, "    unsigned int* s_c%s = readUArrayFromFile(s_c%s_string.c_str());" % (cid,cid)
            print >>cf, "    p.s_c%s = toGPUVector(s_c%s, p.ss, p.ctx, p.queue);" % (cid,cid)
            if kernel == "GaussPoint" or kernel == "GaussRange":
                print >>cf, "    p.bw_c%s = scott_bw(s_c%s,  p.ss, %s);" % (cid,cid,len(kernels))
                print >>cf, "    if(p.bw_c%s < 0.2) p.bw_c%s = 0.2;" % (cid,cid)
            else:
                print >>cf, "    p.bw_c%s = 1.0/(1.0+1.0/%f);" % (cid,dv[cid]-1)  
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
        print >> cf, "    p.local = 64;"
        print >> cf, "    p.global = std::min((size_t) p.ctx.get_device().compute_units()*%s, ((p.ss-1)/p.local+1)*p.local);" % cu_factor
        print >>cf, "    p.gradk = pr.create_kernel(\"grad\");"
        print >>cf, "    p.estk = pr.create_kernel(\"est\");"
        print >>cf, "    p.out = compute::vector<double>(p.global, p.ctx);"
        
        for cid,kernel in enumerate(kernels):
            print >>cf, "    p.grad_c%s = compute::vector<double>(p.global, p.ctx);" % (cid)
        if estimator.bw_optimization == "join":        
            print >>cf, "    std::string true_training_string = iteration_stream.str() + \"/training_join_true.dump\";"
            print >>cf, "    p.training_cardinality = readUArrayFromFile(true_training_string.c_str());"
            print >>cf
            for cid,kernel in enumerate(kernels):
                print >>cf, "    p.grad_c%s = compute::vector<double>(p.ss, p.ctx);" % (cid) 
                if kernel == "GaussRange":
                    print >>cf, "    std::string training_l_c%s_string = iteration_stream.str() + \"/training_join_l_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                    print >>cf, "    p.training_l_c%s= readUArrayFromFile(training_l_c%s_string.c_str());" % (cid,cid)
                    print >>cf, "    std::string training_u_c%s_string = iteration_stream.str() + \"/training_join_u_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                    print >>cf, "    p.training_u_c%s = readUArrayFromFile(training_u_c%s_string.c_str());" % (cid,cid)
                else:
                    print >>cf, "    std::string training_p_c%s_string = iteration_stream.str() + \"/training_join_p_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                    print >>cf, "    p.training_p_c%s = readUArrayFromFile(training_p_c%s_string.c_str());" % (cid,cid) 
                print >>cf
                    
            #Prepare dat bounds
            print >>cf, "    double ub[%s] = {0.0};" % (len(kernels)) 
            print >>cf, "    double lb[%s] = {0.0};" % (len(kernels))
            print >>cf, "    double* bw = (double*) calloc(%s,sizeof(double));" % (len(kernels))
                
            for cid,kernel in enumerate(kernels):
                if kernel == "GaussRange" or kernel == "GaussPoint":
                    print >>cf, "    ub[%s] = fmax(p.bw_c%s,2.0);" % (cid, cid) 
                    print >>cf, "    lb[%s] = 0.001;" % (cid) 
                    print >>cf, "    bw[%s] = 1.0;" % (cid) 
                elif kernel == "CategoricalPoint":
                    print >>cf, "    lb[%s] = DBL_EPSILON;" % (cid) 
                    print >>cf, "    ub[%s] = 1.0-DBL_EPSILON;" % (cid) 
                    print >>cf, "    bw[%s] = 1.0/p.ss;" % (cid) 
    
            print >> cf, "    double minf = 0.0;"
            #print >> cf, "    std::string bwstr(argv[0]);"
            #print >> cf, "    bwstr.append(\".bw_dump\");"
            #print >> cf, "    if(fexists(bwstr.c_str())) bw = readDArrayFromFile(bwstr.c_str());"
                
            for cid,kernel in enumerate(kernels):
                    print >>cf, "    ub[%s] = fmax(bw[%s],ub[%s]);" % (cid, cid,cid) 
                
            print >>cf, """
        nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA,%s);
        nlopt_set_lower_bounds(opt,lb);
        nlopt_set_upper_bounds(opt,ub);
        nlopt_set_maxeval(opt, %s);
        nlopt_set_ftol_rel(opt, %s);
        nlopt_set_ftol_abs(opt, %s);
        nlopt_set_min_objective(opt,obj,&p);
        int frc = nlopt_optimize(opt, bw, &minf);
        assert(frc >=0);    
    
    """ % (len(kernels),1000,"1e-4","1e-3")

    
            #for cid,col in enumerate(kernels):
            #    print >>cf, "    p.bw_c%s = bw[%s];" % (cid,cid)
            #print >> cf, "    ddump(bw, %s, bwstr.c_str());" % len(kernels)
        elif estimator.bw_optimization  == "scott":
            pass
        else:
            raise Exception("GPUKDE does not support this bandwidth optimization.")
        
        print >>cf, "    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";"
        print >>cf, "    p.test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());"

        for cid,kernel in enumerate(kernels):
            if kernel == "GaussRange":
                print >>cf, "    std::string test_l_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                print >>cf, "    p.test_l_c%s= readUArrayFromFile(test_l_c%s_string.c_str());" % (cid,cid)
                print >>cf, "    std::string test_u_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                print >>cf, "    p.test_u_c%s = readUArrayFromFile(test_u_c%s_string.c_str());" % (cid,cid)
            else:
                print >>cf, "    std::string test_p_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid)
                print >>cf, "    p.test_p_c%s = readUArrayFromFile(test_p_c%s_string.c_str());" % (cid,cid) 

        print >>cf
        print >>cf, "    join_test(&p);"
        print >>cf, "}" 
        
        
def generateGPUKDEParameterArray(f,query,estimator,kernels):
    print >>f, """
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
"""
    print >>f, "    unsigned int iteration;"
    print >>f, "    size_t ss;"
    print >>f, "    unsigned int ts;"
    print >>f, "    compute::kernel estk;"
    print >>f, "    compute::kernel gradk;"
    for cid,kernel in enumerate(kernels):
        print >>f, "    compute::vector<unsigned int> s_c%s;" % (cid)
        print >>f, "    compute::vector<double> grad_c%s;" % (cid)
        print >>f, "    double bw_c%s;" % (cid)
     
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "    unsigned int* training_l_c%s;" % (cid)
            print >>f, "    unsigned int* training_u_c%s;" % (cid)
        else:
            print >>f, "    unsigned int* training_p_c%s;" % (cid)
            
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "    unsigned int* test_l_c%s;" % (cid)
            print >>f, "    unsigned int* test_u_c%s;" % (cid)
        else:
            print >>f, "    unsigned int* test_p_c%s;" % (cid) 
            
    print >>f, "    compute::vector<double> out;"  
      
    print >>f, "    unsigned int* training_cardinality;"
    print >>f, "    unsigned int* test_cardinality;"

    print >> f, "    size_t global;"
    print >> f, "    size_t local;"


    print >>f, """
} parameters;
"""   

def generateTableEstGrad(f,tid,query,estimator):
    print >>f, "double est_grad_t%s(parameters* p, double* grad " % tid,
    #Start with computing the invariant contributions 
    for cid,column in enumerate(query.tables[tid].columns):
        if estimator.kernels[tid][cid] == "GaussRange":
            print >>f, "    , unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (tid,cid,tid,cid)
        else:
            print >>f, "    , unsigned int p_t%s_c%s" % (tid,cid)
    print >>f, "){"
    print >>f, "    p->gradk%s.set_args(" % tid
    print >>f, "    ",
    for cid,col in enumerate(query.tables[tid].columns):
        if estimator.kernels[tid][cid] == "GaussRange":
            print >>f, "p->s_t%s_c%s, p->bw_t%s_c%s, u_t%s_c%s, l_t%s_c%s, p->grad_t%s_c%s," % (tid,cid,tid,cid,tid,cid,tid,cid,tid,cid),
        else:
            print >>f, "p->s_t%s_c%s, p->bw_t%s_c%s, p_t%s_c%s, p->grad_t%s_c%s," % (tid,cid,tid,cid,tid,cid,tid,cid),
    print >>f, " p->est_t%s, (unsigned int) p->ss%s );" % (tid,tid)
    print >>f, "    boost::compute::event ev = p->queue.enqueue_nd_range_kernel( p->gradk%s, 1, NULL, &(p->global_t%s), &(p->local_t%s));" % (tid,tid,tid)
    #print >>f, "    ev.wait();"
    print >>f, "    double est = 0.0;"
    #Now, we compute the esitmate
    print >>f, "    boost::compute::reduce(p->est_t%s.begin(), p->est_t%s.begin()+std::min(p->global_t%s,p->ss%s), &est, p->queue);" % (tid,tid,tid,tid)
    for cid,column in enumerate(query.tables[tid].columns):
        print >>f, "    boost::compute::reduce(p->grad_t%s_c%s.begin(), p->grad_t%s_c%s.begin()+std::min(p->global_t%s,p->ss%s), grad+%s, p->queue);" % (tid,cid,tid,cid,tid,tid,cid)

    print >>f, "   p->queue.finish();"
    for cid,column in enumerate(query.tables[tid].columns):
        print >>f, "    *(grad+%s) *= ((double) p->ts%s) /p->ss%s;" % (cid,tid,tid)
    print >> f, "    est *= ((double) p->ts%s)/p->ss%s;" % (tid, tid)
    print >>f, "    return est;"
    print >>f, "}"
    print >>f

def generateTableObjectiveGrad(f,tid,query,estimator):
    print >>f, "double obj_grad_t%s(unsigned n, const double* bw, double* grad, void* f_data){" % tid
    print >>f, "    parameters* p = (parameters*) f_data;"
    print >>f, "    double tmp_grad[%s] = {%s};" % (len(query.tables[tid].columns),','.join(map(str,[0.0]*len(query.tables[tid].columns))))
    for cid,col in enumerate(query.tables[tid].columns):
        print >>f, "    p->bw_t%s_c%s = bw[%s];" % (tid,cid,cid)
        print >>f, "    if(grad != NULL)grad[%s] = 0.0;" % (cid)
    print >>f, "    double objective = 0.0;"
    print >>f, "    double est= 0.0;"
    print >>f, "    int first = 1;"
    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.training
    print >>f, "        if(first ",
    if estimator.kernels[tid][cid] == "GaussRange":
        print >> f, "|| p->u_t%s_c%s[i] != p->u_t%s_c%s[i-1]" % (tid, cid, tid, cid),
        print >> f, "|| p->l_t%s_c%s[i] != p->l_t%s_c%s[i-1]" % (tid, cid, tid, cid),
    else:
        print >> f, "|| p->p_t%s_c%s[i] != p->p_t%s_c%s[i]" % (tid, cid,tid,cid),
    print >> f, "){"
    print >>f, "            first = 0;"
    print >>f, "            est =  est_grad_t%s(p, tmp_grad " % tid,
    for cid,column in enumerate(query.tables[tid].columns):
        if estimator.kernels[tid][cid] == "GaussRange":
            print >>f, ", p->u_t%s_c%s[i], p->l_t%s_c%s[i] " % (tid,cid,tid,cid),
        else:
            print >>f, ", p->p_t%s_c%s[i] " % (tid,cid),
    print >>f, ");"
    print >> f, "        }"
    print >>f, "        unsigned int trues =  p->true_t%s[i];" % tid
    if estimator.objective == "squared":
        print >>f, "        objective += (est-trues)*(est-trues)/%s;" % estimator.training
    elif estimator.objective == "Q":
        print >>f, "        if(est < 1.0) est = 1.0;"
        print >>f, "        objective += std::max(est/trues,trues/est)/%s;" % estimator.training
    else:
        raise Exception("I don't know this objective function.")

    if estimator.objective == "squared":
        for cid,column in enumerate(query.tables[tid].columns):
            print >>f, "        if(grad != NULL) grad[%s] += tmp_grad[%s]*2.0*(est-trues)/%s;" % (cid,cid,estimator.training)
    elif estimator.objective == "Q":
        for cid,column in enumerate(query.tables[tid].columns):
            print >>f, "        if(grad != NULL) grad[%s] += est <=trues ? - tmp_grad[%s]*(trues/(est*est))/%s : (tmp_grad[%s]/trues)/%s;" % (cid,cid,estimator.training,cid,estimator.training)
    else:
        raise Exception("I don't know this objective function.")
    print >>f, "    }"
    print >>f, "    return objective;"
    print >>f, "}"    
    print >>f

def generateGPUKDEEstGrad(f,query,estimator,kernels):
    print >>f, "double est_grad(parameters* p, double* grad ",
    #Start with computing the invariant contributions 
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "    , unsigned int u_c%s, unsigned int l_c%s" % (cid,cid)
        else:
            print >>f, "    , unsigned int p_c%s" % (cid)
    print >>f, "){"
    print >>f, "    p->gradk.set_args("
    print >>f, "    ",
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "p->s_c%s, p->bw_c%s, u_c%s, l_c%s, p->grad_c%s," % (cid,cid,cid,cid,cid),
        else:
            print >>f, "p->s_c%s, p->bw_c%s, p_c%s, p->grad_c%s," % (cid,cid,cid,cid),
    print >>f, " p->out, (unsigned int) p->ss  );"
    print >>f, "    boost::compute::event ev = p->queue.enqueue_nd_range_kernel( p->gradk, 1, NULL, &(p->global), &(p->local));"
    #print >>f, "    ev.wait();"
    print >>f, "    double est = 0.0;"
    #Now, we compute the esitmate
    print >>f, "    boost::compute::reduce(p->out.begin(), p->out.begin()+std::min(p->global, p->ss), &est, p->queue);"
    for cid,kernel in enumerate(kernels):
        print >>f, "    boost::compute::reduce(p->grad_c%s.begin(), p->grad_c%s.begin()+std::min(p->global, p->ss), grad+%s, p->queue);" % (cid,cid,cid)

    print >>f, "    p->queue.finish();"
    print >>f, "    est *= ((double) p->ts)/p->ss;"
    for cid,kernel in enumerate(kernels):
        print >>f, "    *(grad+%s) *= ((double) p->ts) /p->ss;" % (cid)
    print >>f, "    return est;"
    print >>f, "}"
    print >>f



def generateGPUKDEObjectiveGrad(f,query,estimator,kernels):
    print >>f, "double obj_grad(unsigned n, const double* bw, double* grad, void* f_data){"
    print >>f, "    parameters* p = (parameters*) f_data;"
    print >>f, "    double tmp_grad[%s] = {%s};" % (len(kernels),0.0)
    for cid,kernel in enumerate(kernels):
        print >>f, "    p->bw_c%s = bw[%s];" % (cid,cid)
        print >>f, "    if(grad != NULL) grad[%s] = 0.0;" % (cid)
    print >>f, "    double objective = 0.0;"
    print >>f, "    double est = 0.0;"
    print >>f, "    int first = 1;"
    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.training
    print >>f, "       if(first ",
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "|| p->training_u_c%s[i] != p->training_u_c%s[i-1]" % (cid,cid),
            print >>f, "|| p->training_l_c%s[i] != p->training_l_c%s[i-1]" % (cid,cid),
        else:
            print >>f, "|| p->training_p_c%s[i] != p->training_p_c%s[i-1]" % (cid,cid),
    print >>f, "){"
    print >>f, "            first = 0;"
    print >>f, "            est =  est_grad(p, tmp_grad ",
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, ", p->training_u_c%s[i], p->training_l_c%s[i] " % (cid,cid),
        else:
            print >>f, ", p->training_p_c%s[i] " % (cid),
    print >>f, ");"
    print >> f, "        }"
    print >>f, "        unsigned int trues =  p->training_cardinality[i];"

    if estimator.objective == "squared":
        print >>f, "        objective += (est-trues)*(est-trues)/%s;" % estimator.training
    elif estimator.objective == "Q":
        print >>f, "        if(est < 1.0) est = 1.0;"
        print >>f, "        objective += std::max(est/trues,trues/est)/%s;" % estimator.training
    else:
        raise Exception("I don't know this objective function.")

    if estimator.objective == "squared":
        for cid,kernel in enumerate(kernels):
			print >>f, "        if(grad != NULL) grad[%s] += tmp_grad[%s]*2.0*(est-trues)/%s;" % (cid,cid,estimator.training)
    elif estimator.objective == "Q":
        for cid,kernel in enumerate(kernels):
            print >>f, "        if(grad != NULL) grad[%s] += est <=trues ? - tmp_grad[%s]*(trues/(est*est))/%s : (tmp_grad[%s]/trues)/%s;" % (cid,cid,estimator.training,cid,estimator.training)
    else:
        raise Exception("I don't know this objective function.")
    print >>f, "    }"
    print >>f, "    return objective;"
    print >>f, "}"    
    print >>f

def generateGPUKDEObjective(f,query,estimator,kernels):
    print >>f, "double obj(unsigned n, const double* bw, double* grad, void* f_data){"
    print >>f, "    parameters* p = (parameters*) f_data;"
    print >>f, "    double tmp_grad[%s] = {%s};" % (len(kernels),0.0)
    for cid,kernel in enumerate(kernels):
        print >>f, "    p->bw_c%s = bw[%s];" % (cid,cid)
        print >>f, "    if(grad != NULL) grad[%s] = 0.0;" % (cid)

    if estimator.objective == "squared":
        print >>f, "    double objective = 0.0;"
    elif estimator.objective == "Q":
        print >>f, "    double objective = 1.0;"
    else:
        raise Exception("I don't know this objective function.")

    print >>f, "    double est = 0.0;"
    print >>f, "    int first = 1;"
    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.training
    print >>f, "       if(first ",
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "|| p->training_u_c%s[i] != p->training_u_c%s[i-1]" % (cid,cid),
            print >>f, "|| p->training_l_c%s[i] != p->training_l_c%s[i-1]" % (cid,cid),
        else:
            print >>f, "|| p->training_p_c%s[i] != p->training_p_c%s[i-1]" % (cid,cid),
    print >>f, "){"
    print >>f, "            first = 0;"
    print >>f, "            est =  join_estimate_instance(p ",
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, ", p->training_u_c%s[i], p->training_l_c%s[i] " % (cid,cid),
        else:
            print >>f, ", p->training_p_c%s[i] " % (cid),
    print >>f, ");"
    print >> f, "        }"
    print >>f, "        unsigned int trues =  p->training_cardinality[i];"

    if estimator.objective == "squared":
        print >>f, "        objective += (est-trues)*(est-trues)/%s;" % estimator.training
    elif estimator.objective == "Q":
        print >>f, "        if(est < 1.0) est = 1.0;"
        print >>f, "        objective *= std::pow(std::max(est/trues,trues/est),1.0/%s);" % estimator.training
    else:
        raise Exception("I don't know this objective function.")

    print >>f, "    }"
    print >>f, "    return objective;"
    print >>f, "}"    
    print >>f

def generateGPUJKDETestWrapper(f,query,estimator):
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
                if query.tables[i].columns[j].type == "range":
                    print >>f, "|| p->j_l_t%s_c%s[i] != p->j_l_t%s_c%s[i-1]" % (i,j,i,j),
                    print >>f, "|| p->j_u_t%s_c%s[i] != p->j_u_t%s_c%s[i-1]" % (i,j,i,j),
                elif query.tables[i].columns[j].type == "point":
                    print >>f, "|| p->j_p_t%s_c%s[i] != p->j_p_t%s_c%s[i-1] " % (i,j,i,j),
                else:
                    raise Exception("Unknown ctype.")
    print >>f, "){"
    if hasattr(estimator, 'look_behind'):
        if estimator.look_behind:
            print >> f, "            first = 0;"
    else:
        print >>f, "            first = 0;"
    print >>f, "            est = join_estimate_instance(p",                                                       
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        if len(indices) != 0:
            for j in indices:
                if query.tables[i].columns[j].type == "range":
                    print >>f, ", p->j_u_t%s_c%s[i]" % (i,j),
                    print >>f, ", p->j_l_t%s_c%s[i]" % (i,j),
                elif query.tables[i].columns[j].type == "point":
                    print >>f, ", p->j_p_t%s_c%s[i]" % (i,j),
                else:
                    raise Exception("Unknown ctype.")
    print >>f, ");"
    print >>f, "        }"
    print >>f, "        auto end = std::chrono::high_resolution_clock::now();"
    print >>f, "        trues = p->j_test_cardinality[i];"    
    print >>f, "        objective += (est-trues)*(est-trues);" 
    Utils.printObjectiveLine(f,"+".join(map(lambda x : "p->ss%s" % str(x),range(0,len(query.tables)))))
    print >>f, "    }"
    print >>f, "    return objective/%s;" % estimator.test
    print >>f, "}"
    
def generateGPUKDETestWrapper(f,query,estimator,kernels):
    print >>f, "double join_test(parameters* p){"    
    print >>f, "    double objective = 0.0;"
    print >>f, "    double est = 0.0;"
    print >>f, "    int first = 1;"

    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.test
    print >> f, "       auto begin = std::chrono::high_resolution_clock::now();"
    print >>f, "        if(first ",

    for cid, kernel in enumerate(kernels):
        if kernel == "GaussRange":
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
                                                            
    for cid, kernel in enumerate(kernels):
        if kernel == "GaussRange":
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

    
def generateGPUKDEEstimateFunction(f,query,estimator,kernels):
    print >>f, "double join_estimate_instance(parameters* p"
    for cid,kernel in enumerate(kernels):
    #Start with computing the invariant contributions   
        if kernel == "GaussRange":
            print >>f, "    , unsigned int u_c%s, unsigned int l_c%s" % (cid,cid)
        else:
            print >>f, "    , unsigned int p_c%s" % (cid)
    
    print >>f, "){"
    print >>f, "    p->estk.set_args(",
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print >>f, "p->s_c%s, p->bw_c%s, u_c%s, l_c%s, " % (cid,cid,cid,cid),
        else:
            print >>f, "p->s_c%s, p->bw_c%s, p_c%s, " % (cid,cid,cid),
    print >>f, " p->out, (unsigned int) p->ss",
    print >>f, ");"
    print >>f, "    boost::compute::event ev = p->queue.enqueue_nd_range_kernel(p->estk,1,NULL,&(p->global), &(p->local));"
    #print >>f, "    ev.wait();"
    
    print >>f, "    double est = 0.0;"
    print >>f, "    boost::compute::reduce(p->out.begin(), p->out.begin()+std::min(p->global, p->ss), &est, p->queue);"
    print >>f, "    p->queue.finish();"
    print >>f, "    est *= ((double) p->ts)/p->ss;"
    print >>f, "    return est;"
    #At this point, we need a
    print >>f, "}"
