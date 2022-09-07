 #Code generator for gauss kernel
import Utils
local_size = 64

class GaussKernel:
    def rangeEstimateFunction(self,f):
        print("""
T gaussRangeEst(unsigned int v, T h, unsigned int u, unsigned int l){
    if(v == 0){
        return 0.0;
    }
    T up = ((T)u)-v;
    T lo = ((T)l)-v;
    return (erf(up/(M_SQRT2*h)) - erf(lo/(M_SQRT2*h)))*0.5;
}
""", file=f)

    def rangeGradientFunction(self, f ):
        print("""
T gaussRangeGrad(unsigned int v, T h, unsigned int u, unsigned int l){
    if(v == 0){
        return 0.0;
    }
    T up = ((T)u)-v;
    T lo = ((T)l)-v;
    return (lo * exp((T)-1.0 * lo * lo / (2*h*h)) - up * exp((T)-1.0 * up * up / (2*h*h)))*0.5*0.5*M_SQRT2*M_2_SQRTPI/(h*h);
}
""", file=f)

    def rangeGradientConstant(self, f ,bwvar):
        return "M_2_SQRTPI*0.5*M_SQRT2 / (%s*%s)" % (bwvar,bwvar)

    def pointEstimateFunction(self, f ):
        print("""
T gaussPointEst(unsigned int v, T h, unsigned int p){
    if(v == 0){
        return v==p;
    }
    T up = ((T) p)+0.5-v;
    T lo = ((T) p)-0.5-v;
    return (erf(up/(M_SQRT2*h)) - erf(lo/(M_SQRT2*h)))*0.5;
}
""", file=f)

    def pointGradientFunction(self, f ):
        print("""
T gaussPointGrad(unsigned int v, T h, unsigned int p){
    if(v == 0){
        return 0.0;
    }
    T up = ((T) p)+0.5-v;
    T lo = ((T) p)-0.5-v;
    return (lo * exp((T)-1.0 * lo * lo / (2*h*h)) - up * exp((T)-1.0 * up * up / (2*h*h)))*0.5*0.5*M_SQRT2*M_2_SQRTPI/(h*h);
}
""", file=f)


#Code generator for categorical kernel
class CategoricalKernel:
    def pointEstimateFunction(self, f ):
        print("""
T catPointEst(unsigned int v, T h, unsigned int p, unsigned int dvals){
    return (v == p) ? (1.0-h) : h/(dvals-1);
}
""", file=f)

    def pointEstimateConstant(self, f ,bw=None):
        return None

    def pointGradientFunction(self, f ):
        print("""
T catPointGrad(unsigned int v, T h, unsigned int p, unsigned int dvals){
    return (v == p) ? -1.0 : 1.0/(dvals-1);
}
""", file=f)

    def pointGradientConstant(self, f ,bw=None):
        return None
        
def generateTableGradientContributionKernel(f,kname,kernels,dvals = None):
    print("__kernel void %s(" % kname, file=f)
    for i,k in enumerate(kernels):
        if k == "GaussRange":
            print("    __global unsigned int* c%s, T h%s, unsigned int  u%s, unsigned int l%s, __global T* g%s," % (i,i,i,i,i), file=f)
        elif k == "GaussPoint" or k == "CategoricalPoint":
            print("    __global unsigned int* c%s, T h%s, unsigned int p%s, __global T* g%s," % (i,i,i,i), file=f)
        else:
            raise Exception("Unsupported kernel.")
    print("    __global T* o, unsigned int ss){", file=f)
    print("        T cont = 0.0;", file=f)
    for i,k in enumerate(kernels):
        print("        T gcont_c%s = 0.0;" % i, file=f)

    print("        for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){", file=f)
    print("            if (offset + get_global_id(0) < ss){", file=f)
    for i,k in enumerate(kernels):
        if k == "GaussPoint":
            print("                T ec%s = gaussPointEst(c%s[offset+get_global_id(0)], h%s, p%s);" % (i,i,i,i), file=f)
            print("                T gc%s = gaussPointGrad(c%s[offset+get_global_id(0)], h%s, p%s);" % (i,i,i,i), file=f)
        elif k == "GaussRange":
            print("                T ec%s = gaussRangeEst(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (i,i,i,i,i), file=f)
            print("                T gc%s = gaussRangeGrad(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (i,i,i,i,i), file=f)
        elif k == "CategoricalPoint":
            print("                T ec%s = catPointEst(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (i,i,i,i,dvals[i]), file=f)
            print("                T gc%s = catPointGrad(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (i,i,i,i,dvals[i]), file=f)            
        else:
            raise Exception("Unsupported kernel.")
    print("                cont += 1.0 ", end=' ', file=f)
    for i,k in enumerate(kernels):
        print("* ec%s" % i, end=' ', file=f)
    print(";", file=f) 
    for i,_ in enumerate(kernels):
        print("                gcont_c%s += gc%s" % (i,i), end=' ', file=f)
        for j,_ in enumerate(kernels):
            if i == j:
                continue  
            else:
                print("* ec%s" % j, end=' ', file=f)  
        print(";", file=f)
    print("            }", file=f)
    print("        }", file=f)
    print("        if (get_global_id(0) < ss) o[get_global_id(0)]= cont;", file=f)
    for i,_ in enumerate(kernels):
        print("        if (get_global_id(0) < ss) g%s[get_global_id(0)] = gcont_c%s;" % (i,i), file=f)
    print("}", file=f)
    print(file=f)
    
def generateTableEstimateContributionKernel(f,kname,kernels,dvals = None):
    print("__kernel void %s(" % kname, file=f)
    for i,k in enumerate(kernels):
        if k == "GaussRange":
            print("    __global unsigned int* c%s, T h%s, unsigned int  u%s, unsigned int l%s," % (i,i,i,i), file=f)
        elif k == "GaussPoint" or k == "CategoricalPoint":
            print("    __global unsigned int* c%s, T h%s, unsigned int p%s," % (i,i,i), file=f)
        else:
            raise Exception("Unsupported kernel.")
    print("    __global T* o, unsigned int ss){", file=f)
    print("        T cont = 0.0;", file=f)
    print("        for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){", file=f)
    print("            if (offset + get_global_id(0) < ss){", file=f)
    for i,k in enumerate(kernels):
        if k == "GaussPoint":
            print("                T ec%s = gaussPointEst(c%s[offset+get_global_id(0)], h%s, p%s);" % (i,i,i,i), file=f)
        elif k == "GaussRange":
            print("                T ec%s = gaussRangeEst(c%s[offset+get_global_id(0)], h%s, u%s, l%s);" % (i,i,i,i,i), file=f)
        elif k == "CategoricalPoint":
            print("                T ec%s = catPointEst(c%s[offset+get_global_id(0)], h%s, p%s,%s);" % (i,i,i,i,dvals[i]), file=f)
        else:
            raise Exception("Unsupported kernel.")
    print("                cont += 1.0 ", end=' ', file=f)
    for i,k in enumerate(kernels):
        print("* ec%s" % i, end=' ', file=f)
    print(";", file=f)
    print("            }", file=f)
    print("        }", file=f)
    print("        if (get_global_id(0) < ss) o[get_global_id(0)] = cont;", file=f)
    print("}", file=f)
    print(file=f)


def generatePreamble(f):
    print("""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef M_SQRT2
     #define M_SQRT2     1.41421356237309504880168872420969808
#endif
typedef double T;
""", file=f)
    
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
def generateGPUJKDELocalTraining(cf,query,estimator, cu_factor):
            raise Exception("This code needs to be checked and should not be used right now.")
            for tid,t in enumerate(query.tables):
                #print >>cf, "    p.estk%s = pr.create_kernel(\"est_t%s\");" % (tid,tid)   
                print("    p.local_t%s = 64;" % tid, file=cf)
                print("    p.global_t%s = std::min((size_t) p.ctx.get_device().compute_units()*%s , ((p.ss%s-1)/p.local_t%s+1)*p.local_t%s);" % (tid,cu_factortid,tid,tid), file=cf)
                print("    p.localt_t%s = std::min(p.local_t%s,p.global_t%s);" % (tid, tid, tid), file=cf)
                print("    p.gradk%s = pr.create_kernel(\"grad_t%s\");" % (tid,tid), file=cf)   
                print("    p.est_t%s = compute::vector<double>(p.global_t%s, p.ctx);" % (tid,tid), file=cf) 
                print("    std::string true_t%s_string = iteration_stream.str() + \"/training_table_%s_true.dump\";" % (tid,query.tables[tid].tid), file=cf)
                print("    p.true_t%s = readUArrayFromFile(true_t%s_string.c_str());" % (tid,tid), file=cf)
                print(file=cf)
                for cid,col in enumerate(t.columns):
                    print("    p.grad_t%s_c%s = compute::vector<double>(p.global_t%s, p.ctx);" % (tid,cid,tid), file=cf) 
                    if estimator.kernels[tid][cid] == "GaussRange":
                        print("    std::string l_t%s_c%s_string = iteration_stream.str() + \"/training_table_l_%s_%s.dump\";" % (tid,cid,query.tables[tid].tid,query.tables[tid].columns[cid].cid), file=cf)
                        print("    p.l_t%s_c%s= readUArrayFromFile(l_t%s_c%s_string.c_str());" % (tid,cid), file=cf)
                        print("    std::string u_t%s_c%s_string = iteration_stream.str() + \"/training_table_u_%s_%s.dump\";" % (tid,cid,query.tables[tid].tid,query.tables[tid].columns[cid].cid), file=cf)
                        print("    p.u_t%s_c%s = readUArrayFromFile(u_t%s_c%s_string.c_str());" % (tid,cid), file=cf)
                    else:
                        print("    std::string p_t%s_c%s_string = iteration_stream.str() + \"/training_table_p_%s_%s.dump\";" % (tid,cid,query.tables[tid].tid,query.tables[tid].columns[cid].cid), file=cf)
                        print("    p.p_t%s_c%s = readUArrayFromFile(p_t%s_c%s_string.c_str());" % (tid,cid,tid,cid), file=cf) 
                    print(file=cf)
                    
                #Prepare dat bounds
                print("    double ub%s[%s] = {0.0};" % (tid,len(t.columns)), file=cf) 
                print("    double lb%s[%s] = {0.0};" % (tid,len(t.columns)), file=cf) 
                
                print("    double minf%s = 0.0;" % tid, file=cf)
                print("    double bw%s[%s] = {0.0};" % (tid,len(t.columns)), file=cf)
                for cid,col in enumerate(t.columns):
                    print("    bw%s[%s] = p.bw_t%s_c%s;" % (tid,cid,tid,cid), file=cf)
                    
                for cid,col in enumerate(t.columns):
                    if estimator.kernels[tid][cid] == "GaussRange" or estimator.kernels[tid][cid] == "GaussPoint":
                        print("    ub%s[%s] = p.bw_t%s_c%s;" % (tid,cid, tid,cid), file=cf) 
                        print("    lb%s[%s] = 0.1;" % (tid,cid), file=cf) 
                    elif estimator.kernels[tid][cid] == "CategoricalPoint":
                        print("    lb%s[%s] = DBL_EPSILON;" % (tid,cid), file=cf) 
                        print("    ub%s[%s] = 1.0-DBL_EPSILON;" % (tid,cid), file=cf)
                        print("    bw%s[%s] = 1.0/p.ss%s;" % (tid,cid,tid), file=cf) 
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
                 
                print("""
    nlopt_opt opt%s = nlopt_create(NLOPT_LD_MMA,%s);
    nlopt_set_lower_bounds(opt%s,lb%s);
    nlopt_set_upper_bounds(opt%s,ub%s);
    nlopt_set_maxeval(opt%s, %s);
    nlopt_set_ftol_rel(opt%s, %s);
    nlopt_set_min_objective(opt%s,obj_grad_t%s,&p);
    int frc%s = nlopt_optimize(opt%s, bw%s, &minf%s);
    assert(frc%s >=0);    
""" % (tid,len(t.columns),tid,tid,tid,tid,tid,1000,tid,"1e-5",tid,tid,tid,tid,tid,tid,tid), file=cf)
                for cid,col in enumerate(t.columns):
                    print("    p.bw_t%s_c%s = bw%s[%s];" % (tid,cid,tid,cid), file=cf)
                print(file=cf)
                    
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
        print("    p.ts = %s;" % (ts), file=cf)
        for cid,kernel in enumerate(kernels):
            print("    std::stringstream s_c%s_stream ;" % (cid), file=cf)
            print("    s_c%s_stream << iteration_stream.str() << \"/jsample_\" << atoi(argv[1]) << \"_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf) 
            print("    std::string s_c%s_string = s_c%s_stream.str();" % (cid,cid), file=cf)
            print("    unsigned int* s_c%s = readUArrayFromFile(s_c%s_string.c_str());" % (cid,cid), file=cf)
            print("    p.s_c%s = toGPUVector(s_c%s, p.ss, p.ctx, p.queue);" % (cid,cid), file=cf)
            if kernel == "GaussPoint" or kernel == "GaussRange":
                print("    p.bw_c%s = scott_bw(s_c%s,  p.ss, %s);" % (cid,cid,len(kernels)), file=cf)
                print("    if(p.bw_c%s < 0.2) p.bw_c%s = 0.2;" % (cid,cid), file=cf)
            else:
                print("    p.bw_c%s = 1.0/(1.0+1.0/%f);" % (cid,dv[cid]-1), file=cf)  
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
        print("    p.local = 64;", file=cf)
        print("    p.global = std::min((size_t) p.ctx.get_device().compute_units()*%s, ((p.ss-1)/p.local+1)*p.local);" % cu_factor, file=cf)
        print("    p.local = std::min(p.global, p.local);", file=cf)
        print("    p.gradk = pr.create_kernel(\"grad\");", file=cf)
        print("    p.estk = pr.create_kernel(\"est\");", file=cf)
        print("    p.out = compute::vector<double>(p.global, p.ctx);", file=cf)
        
        for cid,kernel in enumerate(kernels):
            print("    p.grad_c%s = compute::vector<double>(p.global, p.ctx);" % (cid), file=cf)
        if estimator.bw_optimization == "join":        
            print("    std::string true_training_string = iteration_stream.str() + \"/training_join_true.dump\";", file=cf)
            print("    p.training_cardinality = readUArrayFromFile(true_training_string.c_str());", file=cf)
            print(file=cf)
            for cid,kernel in enumerate(kernels):
                print("    p.grad_c%s = compute::vector<double>(p.ss, p.ctx);" % (cid), file=cf) 
                if kernel == "GaussRange":
                    print("    std::string training_l_c%s_string = iteration_stream.str() + \"/training_join_l_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                    print("    p.training_l_c%s= readUArrayFromFile(training_l_c%s_string.c_str());" % (cid,cid), file=cf)
                    print("    std::string training_u_c%s_string = iteration_stream.str() + \"/training_join_u_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                    print("    p.training_u_c%s = readUArrayFromFile(training_u_c%s_string.c_str());" % (cid,cid), file=cf)
                else:
                    print("    std::string training_p_c%s_string = iteration_stream.str() + \"/training_join_p_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                    print("    p.training_p_c%s = readUArrayFromFile(training_p_c%s_string.c_str());" % (cid,cid), file=cf) 
                print(file=cf)
                    
            #Prepare dat bounds
            print("    double ub[%s] = {0.0};" % (len(kernels)), file=cf) 
            print("    double lb[%s] = {0.0};" % (len(kernels)), file=cf)
            print("    double* bw = (double*) calloc(%s,sizeof(double));" % (len(kernels)), file=cf)
                
            for cid,kernel in enumerate(kernels):
                if kernel == "GaussRange" or kernel == "GaussPoint":
                    print("    ub[%s] = fmax(p.bw_c%s,2.0);" % (cid, cid), file=cf) 
                    print("    lb[%s] = 0.001;" % (cid), file=cf) 
                    print("    bw[%s] = 1.0;" % (cid), file=cf) 
                elif kernel == "CategoricalPoint":
                    print("    lb[%s] = DBL_EPSILON;" % (cid), file=cf) 
                    print("    ub[%s] = 1.0-DBL_EPSILON;" % (cid), file=cf) 
                    print("    bw[%s] = 1.0/p.ss;" % (cid), file=cf) 
    
            print("    double minf = 0.0;", file=cf)
            #print >> cf, "    std::string bwstr(argv[0]);"
            #print >> cf, "    bwstr.append(\".bw_dump\");"
            #print >> cf, "    if(fexists(bwstr.c_str())) bw = readDArrayFromFile(bwstr.c_str());"
                
            for cid,kernel in enumerate(kernels):
                    print("    ub[%s] = fmax(bw[%s],ub[%s]);" % (cid, cid,cid), file=cf) 
                
            print("""
        nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA,%s);
        nlopt_set_lower_bounds(opt,lb);
        nlopt_set_upper_bounds(opt,ub);
        nlopt_set_maxeval(opt, %s);
        nlopt_set_ftol_rel(opt, %s);
        nlopt_set_ftol_abs(opt, %s);
        nlopt_set_min_objective(opt,obj,&p);
        int frc = nlopt_optimize(opt, bw, &minf);
        assert(frc >=0);    
    
    """ % (len(kernels),1000,"1e-4","1e-3"), file=cf)
            for cid,kernel in enumerate(kernels):
                print("    p.bw_c%s = bw[%s];" % (cid,cid), file=cf)

    
            #for cid,col in enumerate(kernels):
            #    print >>cf, "    p.bw_c%s = bw[%s];" % (cid,cid)
            #print >> cf, "    ddump(bw, %s, bwstr.c_str());" % len(kernels)
        elif estimator.bw_optimization  == "scott":
            pass
        else:
            raise Exception("GPUKDE does not support this bandwidth optimization.")
        
        print("    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";", file=cf)
        print("    p.test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());", file=cf)

        for cid,kernel in enumerate(kernels):
            if kernel == "GaussRange":
                print("    std::string test_l_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                print("    p.test_l_c%s= readUArrayFromFile(test_l_c%s_string.c_str());" % (cid,cid), file=cf)
                print("    std::string test_u_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                print("    p.test_u_c%s = readUArrayFromFile(test_u_c%s_string.c_str());" % (cid,cid), file=cf)
            else:
                print("    std::string test_p_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (cid,query.tables[remap[cid][0]].tid,query.tables[remap[cid][0]].columns[remap[cid][1]].cid), file=cf)
                print("    p.test_p_c%s = readUArrayFromFile(test_p_c%s_string.c_str());" % (cid,cid), file=cf) 

        print(file=cf)
        print("    join_test(&p);", file=cf)
        print("}", file=cf) 
        
        
def generateGPUKDEParameterArray(f,query,estimator,kernels):
    print("""
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
""", file=f)
    print("    unsigned int iteration;", file=f)
    print("    size_t ss;", file=f)
    print("    unsigned int ts;", file=f)
    print("    compute::kernel estk;", file=f)
    print("    compute::kernel gradk;", file=f)
    for cid,kernel in enumerate(kernels):
        print("    compute::vector<unsigned int> s_c%s;" % (cid), file=f)
        print("    compute::vector<double> grad_c%s;" % (cid), file=f)
        print("    double bw_c%s;" % (cid), file=f)
     
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print("    unsigned int* training_l_c%s;" % (cid), file=f)
            print("    unsigned int* training_u_c%s;" % (cid), file=f)
        else:
            print("    unsigned int* training_p_c%s;" % (cid), file=f)
            
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print("    unsigned int* test_l_c%s;" % (cid), file=f)
            print("    unsigned int* test_u_c%s;" % (cid), file=f)
        else:
            print("    unsigned int* test_p_c%s;" % (cid), file=f) 
            
    print("    compute::vector<double> out;", file=f)  
      
    print("    unsigned int* training_cardinality;", file=f)
    print("    unsigned int* test_cardinality;", file=f)

    print("    size_t global;", file=f)
    print("    size_t local;", file=f)


    print("""
} parameters;
""", file=f)   

def generateTableEstGrad(f,tid,query,estimator):
    print("double est_grad_t%s(parameters* p, double* grad " % tid, end=' ', file=f)
    #Start with computing the invariant contributions 
    for cid,column in enumerate(query.tables[tid].columns):
        if estimator.kernels[tid][cid] == "GaussRange":
            print("    , unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (tid,cid,tid,cid), file=f)
        else:
            print("    , unsigned int p_t%s_c%s" % (tid,cid), file=f)
    print("){", file=f)
    print("    p->gradk%s.set_args(" % tid, file=f)
    print("    ", end=' ', file=f)
    for cid,col in enumerate(query.tables[tid].columns):
        if estimator.kernels[tid][cid] == "GaussRange":
            print("p->s_t%s_c%s, p->bw_t%s_c%s, u_t%s_c%s, l_t%s_c%s, p->grad_t%s_c%s," % (tid,cid,tid,cid,tid,cid,tid,cid,tid,cid), end=' ', file=f)
        else:
            print("p->s_t%s_c%s, p->bw_t%s_c%s, p_t%s_c%s, p->grad_t%s_c%s," % (tid,cid,tid,cid,tid,cid,tid,cid), end=' ', file=f)
    print(" p->est_t%s, (unsigned int) p->ss%s );" % (tid,tid), file=f)
    print("    boost::compute::event ev = p->queue.enqueue_nd_range_kernel( p->gradk%s, 1, NULL, &(p->global_t%s), &(p->local_t%s));" % (tid,tid,tid), file=f)
    #print >>f, "    ev.wait();"
    print("    double est = 0.0;", file=f)
    #Now, we compute the esitmate
    print("    boost::compute::reduce(p->est_t%s.begin(), p->est_t%s.begin()+std::min(p->global_t%s,p->ss%s), &est, p->queue);" % (tid,tid,tid,tid), file=f)
    for cid,column in enumerate(query.tables[tid].columns):
        print("    boost::compute::reduce(p->grad_t%s_c%s.begin(), p->grad_t%s_c%s.begin()+std::min(p->global_t%s,p->ss%s), grad+%s, p->queue);" % (tid,cid,tid,cid,tid,tid,cid), file=f)

    print("   p->queue.finish();", file=f)
    for cid,column in enumerate(query.tables[tid].columns):
        print("    *(grad+%s) *= ((double) p->ts%s) /p->ss%s;" % (cid,tid,tid), file=f)
    print("    est *= ((double) p->ts%s)/p->ss%s;" % (tid, tid), file=f)
    print("    return est;", file=f)
    print("}", file=f)
    print(file=f)

def generateTableObjectiveGrad(f,tid,query,estimator):
    print("double obj_grad_t%s(unsigned n, const double* bw, double* grad, void* f_data){" % tid, file=f)
    print("    parameters* p = (parameters*) f_data;", file=f)
    print("    double tmp_grad[%s] = {%s};" % (len(query.tables[tid].columns),','.join(map(str,[0.0]*len(query.tables[tid].columns)))), file=f)
    for cid,col in enumerate(query.tables[tid].columns):
        print("    p->bw_t%s_c%s = bw[%s];" % (tid,cid,cid), file=f)
        print("    if(grad != NULL)grad[%s] = 0.0;" % (cid), file=f)
    print("    double objective = 0.0;", file=f)
    print("    double est= 0.0;", file=f)
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.training, file=f)
    print("            est =  est_grad_t%s(p, tmp_grad " % tid, end=' ', file=f)
    for cid,column in enumerate(query.tables[tid].columns):
        if estimator.kernels[tid][cid] == "GaussRange":
            print(", p->u_t%s_c%s[i], p->l_t%s_c%s[i] " % (tid,cid,tid,cid), end=' ', file=f)
        else:
            print(", p->p_t%s_c%s[i] " % (tid,cid), end=' ', file=f)
    print(");", file=f)
    print("        unsigned int trues =  p->true_t%s[i];" % tid, file=f)
    if estimator.objective == "squared":
        print("        objective += (est-trues)*(est-trues)/%s;" % estimator.training, file=f)
    elif estimator.objective == "Q":
        print("        if(est < 1.0) est = 1.0;", file=f)
        print("        objective += std::max(est/trues,trues/est)/%s;" % estimator.training, file=f)
    else:
        raise Exception("I don't know this objective function.")

    if estimator.objective == "squared":
        for cid,column in enumerate(query.tables[tid].columns):
            print("        if(grad != NULL) grad[%s] += tmp_grad[%s]*2.0*(est-trues)/%s;" % (cid,cid,estimator.training), file=f)
    elif estimator.objective == "Q":
        for cid,column in enumerate(query.tables[tid].columns):
            print("        if(grad != NULL) grad[%s] += est <=trues ? - tmp_grad[%s]*(trues/(est*est))/%s : (tmp_grad[%s]/trues)/%s;" % (cid,cid,estimator.training,cid,estimator.training), file=f)
    else:
        raise Exception("I don't know this objective function.")
    print("    }", file=f)
    print("    return objective;", file=f)
    print("}", file=f)    
    print(file=f)

def generateGPUKDEEstGrad(f,query,estimator,kernels):
    print("double est_grad(parameters* p, double* grad ", end=' ', file=f)
    #Start with computing the invariant contributions 
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print("    , unsigned int u_c%s, unsigned int l_c%s" % (cid,cid), file=f)
        else:
            print("    , unsigned int p_c%s" % (cid), file=f)
    print("){", file=f)
    print("    p->gradk.set_args(", file=f)
    print("    ", end=' ', file=f)
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print("p->s_c%s, p->bw_c%s, u_c%s, l_c%s, p->grad_c%s," % (cid,cid,cid,cid,cid), end=' ', file=f)
        else:
            print("p->s_c%s, p->bw_c%s, p_c%s, p->grad_c%s," % (cid,cid,cid,cid), end=' ', file=f)
    print(" p->out, (unsigned int) p->ss  );", file=f)
    print("    boost::compute::event ev = p->queue.enqueue_nd_range_kernel( p->gradk, 1, NULL, &(p->global), &(p->local));", file=f)
    #print >>f, "    ev.wait();"
    print("    double est = 0.0;", file=f)
    #Now, we compute the esitmate
    print("    boost::compute::reduce(p->out.begin(), p->out.begin()+std::min(p->global, p->ss), &est, p->queue);", file=f)
    for cid,kernel in enumerate(kernels):
        print("    boost::compute::reduce(p->grad_c%s.begin(), p->grad_c%s.begin()+std::min(p->global, p->ss), grad+%s, p->queue);" % (cid,cid,cid), file=f)

    print("    p->queue.finish();", file=f)
    print("    est *= ((double) p->ts)/p->ss;", file=f)
    for cid,kernel in enumerate(kernels):
        print("    *(grad+%s) *= ((double) p->ts) /p->ss;" % (cid), file=f)
    print("    return est;", file=f)
    print("}", file=f)
    print(file=f)



def generateGPUKDEObjectiveGrad(f,query,estimator,kernels):
    print("double obj_grad(unsigned n, const double* bw, double* grad, void* f_data){", file=f)
    print("    parameters* p = (parameters*) f_data;", file=f)
    print("    double tmp_grad[%s] = {%s};" % (len(kernels),0.0), file=f)
    for cid,kernel in enumerate(kernels):
        print("    p->bw_c%s = bw[%s];" % (cid,cid), file=f)
        print("    if(grad != NULL) grad[%s] = 0.0;" % (cid), file=f)
    print("    double objective = 0.0;", file=f)
    print("    double est = 0.0;", file=f)
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.training, file=f)
    print("            est =  est_grad(p, tmp_grad ", end=' ', file=f)
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print(", p->training_u_c%s[i], p->training_l_c%s[i] " % (cid,cid), end=' ', file=f)
        else:
            print(", p->training_p_c%s[i] " % (cid), end=' ', file=f)
    print(");", file=f)
    print("        unsigned int trues =  p->training_cardinality[i];", file=f)

    if estimator.objective == "squared":
        print("        objective += (est-trues)*(est-trues)/%s;" % estimator.training, file=f)
    elif estimator.objective == "Q":
        print("        if(est < 1.0) est = 1.0;", file=f)
        print("        objective += std::max(est/trues,trues/est)/%s;" % estimator.training, file=f)
    else:
        raise Exception("I don't know this objective function.")

    if estimator.objective == "squared":
        for cid,kernel in enumerate(kernels):
            print("        if(grad != NULL) grad[%s] += tmp_grad[%s]*2.0*(est-trues)/%s;" % (cid,cid,estimator.training), file=f)
    elif estimator.objective == "Q":
        for cid,kernel in enumerate(kernels):
            print("        if(grad != NULL) grad[%s] += est <=trues ? - tmp_grad[%s]*(trues/(est*est))/%s : (tmp_grad[%s]/trues)/%s;" % (cid,cid,estimator.training,cid,estimator.training), file=f)
    else:
        raise Exception("I don't know this objective function.")
    print("    }", file=f)
    print("    return objective;", file=f)
    print("}", file=f)    
    print(file=f)

def generateGPUKDEObjective(f,query,estimator,kernels):
    print("double obj(unsigned n, const double* bw, double* grad, void* f_data){", file=f)
    print("    parameters* p = (parameters*) f_data;", file=f)
    print("    double tmp_grad[%s] = {%s};" % (len(kernels),0.0), file=f)
    for cid,kernel in enumerate(kernels):
        print("    p->bw_c%s = bw[%s];" % (cid,cid), file=f)
        print("    if(grad != NULL) grad[%s] = 0.0;" % (cid), file=f)

    if estimator.objective == "squared":
        print("    double objective = 0.0;", file=f)
    elif estimator.objective == "Q":
        print("    double objective = 1.0;", file=f)
    else:
        raise Exception("I don't know this objective function.")

    print("    double est = 0.0;", file=f)
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.training, file=f)
    print("            est =  join_estimate_instance(p ", end=' ', file=f)
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print(", p->training_u_c%s[i], p->training_l_c%s[i] " % (cid,cid), end=' ', file=f)
        else:
            print(", p->training_p_c%s[i] " % (cid), end=' ', file=f)
    print(");", file=f)
    print("        unsigned int trues =  p->training_cardinality[i];", file=f)

    if estimator.objective == "squared":
        print("        objective += (est-trues)*(est-trues)/%s;" % estimator.training, file=f)
    elif estimator.objective == "Q":
        print("        if(est < 1.0) est = 1.0;", file=f)
        print("        objective *= std::pow(std::max(est/trues,trues/est),1.0/%s);" % estimator.training, file=f)
    else:
        raise Exception("I don't know this objective function.")

    print("    }", file=f)
    print("    return objective;", file=f)
    print("}", file=f)    
    print(file=f)

def generateGPUJKDETestWrapper(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    #print >>f, "double join_test(unsigned n, const double* bw, double* grad, void* f_data){"
    print("double join_test(parameters* p){", file=f)    
    #print >>f, "    parameters* p = (parameters*) f_data;"
    print("    double objective = 0.0;", file=f)
    print("    double trues = 0.0;", file=f) 
    print("    double est = 0.0;", file=f) 
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.test, file=f)
    print("        auto begin = std::chrono::high_resolution_clock::now();", file=f)
    print("            est = join_estimate_instance(p", end=' ', file=f)                                                       
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        if len(indices) != 0:
            for j in indices:
                if query.tables[i].columns[j].type == "range":
                    print(", p->j_u_t%s_c%s[i]" % (i,j), end=' ', file=f)
                    print(", p->j_l_t%s_c%s[i]" % (i,j), end=' ', file=f)
                elif query.tables[i].columns[j].type == "point":
                    print(", p->j_p_t%s_c%s[i]" % (i,j), end=' ', file=f)
                else:
                    raise Exception("Unknown ctype.")
    print(");", file=f)
    print("        auto end = std::chrono::high_resolution_clock::now();", file=f)
    print("        trues = p->j_test_cardinality[i];", file=f)    
    print("        objective += (est-trues)*(est-trues);", file=f) 
    Utils.printObjectiveLine(f,"+".join(["p->ss%s" % str(x) for x in range(0,len(query.tables))]))
    print("    }", file=f)
    print("    return objective/%s;" % estimator.test, file=f)
    print("}", file=f)
    
def generateGPUKDETestWrapper(f,query,estimator,kernels):
    print("double join_test(parameters* p){", file=f)    
    print("    double objective = 0.0;", file=f)
    print("    double est = 0.0;", file=f)
    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.test, file=f)
    print("       auto begin = std::chrono::high_resolution_clock::now();", file=f)
    print("            est = join_estimate_instance(p", end=' ', file=f)
                                                            
    for cid, kernel in enumerate(kernels):
        if kernel == "GaussRange":
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

    
def generateGPUKDEEstimateFunction(f,query,estimator,kernels):
    print("double join_estimate_instance(parameters* p", file=f)
    for cid,kernel in enumerate(kernels):
    #Start with computing the invariant contributions   
        if kernel == "GaussRange":
            print("    , unsigned int u_c%s, unsigned int l_c%s" % (cid,cid), file=f)
        else:
            print("    , unsigned int p_c%s" % (cid), file=f)
    
    print("){", file=f)
    print("    p->estk.set_args(", end=' ', file=f)
    for cid,kernel in enumerate(kernels):
        if kernel == "GaussRange":
            print("p->s_c%s, p->bw_c%s, u_c%s, l_c%s, " % (cid,cid,cid,cid), end=' ', file=f)
        else:
            print("p->s_c%s, p->bw_c%s, p_c%s, " % (cid,cid,cid), end=' ', file=f)
    print(" p->out, (unsigned int) p->ss", end=' ', file=f)
    print(");", file=f)
    print("    boost::compute::event ev = p->queue.enqueue_nd_range_kernel(p->estk,1,NULL,&(p->global), &(p->local));", file=f)
    #print >>f, "    ev.wait();"
    
    print("    double est = 0.0;", file=f)
    print("    boost::compute::reduce(p->out.begin(), p->out.begin()+std::min(p->global, p->ss), &est, p->queue);", file=f)
    print("    p->queue.finish();", file=f)
    print("    est *= ((double) p->ts)/p->ss;", file=f)
    print("    return est;", file=f)
    #At this point, we need a
    print("}", file=f)
