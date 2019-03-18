 #Code generator for gauss kernel
import Utils
local_size = 64
from KDECodeGenerator import GaussKernel
from KDECodeGenerator import CategoricalKernel
import copy
from KDECodeGenerator import generateTableGradientContributionKernel
from KDECodeGenerator import generateTableEstGrad
from KDECodeGenerator import generateTableObjectiveGrad
from KDECodeGenerator import generatePreamble
from KDECodeGenerator import generateGPUJKDETestWrapper
from KDECodeGenerator import generateGPUJKDELocalTraining
from JoinGraph import constructJoinGraph
import operator
        
def prod(iterable):
    return reduce(operator.mul, iterable, 1)        
        
def generateBinarySearchCode(f):
    print >>f, """
unsigned int binarySearch(__global unsigned int* x, double val,unsigned int len){
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
    
def generateRectKDELimit(f):
    print >>f, """
    double compute_rect_limit(double bw1, double bw2){
        bw1 = fmax(1.0,bw1 * 3.07634);
        bw2 = fmax(1.0,bw2 * 3.07634);
        double obw1 = 2* ((unsigned int) (bw1+1.0)/2)+1;
        double obw2 = 2* ((unsigned int) (bw2+1.0)/2)+1;
        return (obw1+obw2)/2;
    }
    """

def generateContKDELimit(f):
    print >>f, """
    double compute_cont_limit(double bw1, double bw2, double t1, double t2){
        double bw_term = bw1*bw1 + bw2*bw2;
        return  sqrt(-2.0 * bw_term * log(sqrt(2*M_PI*bw_term)/ (t1*t2)));
    }
    """

class CatKDEKernel:
    def generateEstimateCode(self,f, query,base,node,stats):
        ts,dvals = stats
        t1,c1 = node.left_col
        t2, c2 = node.right_col

        icols = Utils.generateInvariantColumns(query)
        print >> f, "%sT j = (1-bw_t%s_c%s)*(1-bw_t%s_c%s) +  bw_t%s_c%s * bw_t%s_c%s / (%f-1.0);" % ("    " * base,t1,c1,t2,c2,t1,c1,t2,c2,max(dvals[t1][c1],dvals[t2][c2]))

        print >> f, "%ssum += j " % ("    " * base),
        if icols[t1]:
            print >> f, "* c_t%s" % (t1),
        if icols[t2]:
            print >> f, "* c_t%s" % (t2),
        print >> f, ";"
        if icols[t2]:
            print >> f, "%sosum += c_t%s;" % ("    " * base,t2)

    def generateCountCode(self,f, query,base,node,stats):
        print >> f, "%ssum += 1.0;" % ("    " * base)


    def generatePreamble(self,f,query):
        print >> f, "    T sum = 0.0;"
        print >> f, "    T osum = 0.0;"

   
class ContKDEKernel:
    def generateEstimateCode(self,f, query,base,node,stats):
        if len(query.joins) == 1 and len(query.joins[0]) == 2:
            self.generate2EstimateCode(f, query,base,node,stats)
            return
            
        icols = Utils.generateInvariantColumns(query)
        for j, join in enumerate(query.joins):
            print >> f, "%sunsigned int min_val_j%s = 0;" % ("    " * (base), j)
            for pt in join:
                a, b = pt
                print >> f, "%smin_val_j%s = min(min_val_j%s,val_t%s_c%s);" % ("    " * (base), j, j, a, b)
            for pt in join:
                a, b = pt
                print >> f, "%sunsigned int sval_t%s_c%s = val_t%s_c%s-min_val_j%s;" % ("    " * (base),a,b,a,b,j)

            print >> f, "%sT mu_sum_j%s = (0.0 " % ("    " * (base), j),
            for pt in join:
                a, b = pt
                print >> f, "+ sval_t%s_c%s / bw_t%s_c%s" % (a, b, a, b),
            print >> f, ") / sigma_sum_inv_j%s;" % j

            print >> f, "%sT scaled_mu_sum_j%s = 0.0" % ("    " * base, j),
            for pt in join:
                a, b = pt
                print >> f, "+ ((T)sval_t%s_c%s)*sval_t%s_c%s / bw_t%s_c%s" % (a, b, a, b, a, b),
            print >> f, ";"

            print >> f, "%sT j%s = exp(-0.5 *(scaled_mu_sum_j%s - mu_sum_j%s * mu_sum_j%s * sigma_sum_inv_j%s));" % (
            "    " * base, j, j, j, j, j)

        print >> f, "%ssum += 1.0 " % ("    " * base),
        for j, join in enumerate(query.joins):
            print >> f, "* j%s * factor_j%s * pow(0.5*M_1_PI,%s)" % (j,j,(len(join)-1)/2.0),
        for i,c in enumerate(icols):
            if len(c) != 0:
                print >> f, "* c_t%s" % (i),
        print >> f, ";"

    def generate2EstimateCode(self,f, query,base,node,stats):
        icols = Utils.generateJoinColumns(query)
        if len(query.joins) != 1:
            raise Exception("This feature was disabled due to huge ass numerical instabilities.")

        join = query.joins[0]
        t1,c1 = join[0]
        t2,c2 = join[1]

        print >> f, "%ssum += M_SQRT1_2 * M_2_SQRTPI * 0.5 / resbw * exp( -0.5 * (((T)val_t%s_c%s)-((T)val_t%s_c%s))*(((T)val_t%s_c%s)-((T)val_t%s_c%s))/(resbw*resbw)) " % ("    " * base,t1,c1,t2,c2,t1,c1,t2,c2),
        for i, pt in enumerate(join):
            a, b = pt
            print >> f, "* c_t%s" % (a),
        print >> f, ";"

    def generateCountCode(self,f, query,base,node,stats):
        print >> f, "%ssum += 1.0;" % ("    " * base),


    def generatePreamble(self,f,query):
        if len(query.joins) == 1 and len(query.joins[0]) == 2:
            self.generate2Preamble(f, query)
            return

        print >> f, "    T sum = 0.0;"
        # Create all
        for x, join in enumerate(query.joins):
            for pt in join:
                a, b = pt
                print >> f, "    bw_t%s_c%s *= bw_t%s_c%s;" % (a, b, a, b)
        print >> f

        for x, join in enumerate(query.joins):
            print >> f, "    T sigma_prod_j%s = 1.0" % x,
            for pt in join:
                a, b = pt
                print >> f, "* bw_t%s_c%s" % (a, b),
            print >> f, ";"

        for x, join in enumerate(query.joins):
            print >> f, "    T sigma_sum_inv_j%s = 0.0" % x,
            for pt in join:
                a, b = pt
                print >> f, "+ 1.0/(bw_t%s_c%s)" % (a, b),
            print >> f, ";"

            print >> f
            print >> f, "    T factor_j%s = sqrt(1.0/(sigma_prod_j%s*sigma_sum_inv_j%s));" % (x, x, x)


    def generate2Preamble(self,f,query):
        print >> f, "    T sum = 0.0;"
        print >> f, "    T resbw = sqrt(0.0 + "
        # Create all
        for x, join in enumerate(query.joins):
            for pt in join:
                a, b = pt
                print >> f, " + bw_t%s_c%s * bw_t%s_c%s" % (a, b, a, b),
        print >> f, ");"
        print >> f

class RectKDEKernel:
    def upper_bound(self, cols):
        a, b = cols.pop()
        if cols:
            return "min(val_t%s_c%s + ibw_t%s_c%s/2.0, %s)" % (a, b, a, b, self.upper_bound(cols))
        else:
            return "val_t%s_c%s + ibw_t%s_c%s/2.0" % (a, b, a, b)

    def lower_bound(self, cols):
        a, b = cols.pop()
        if cols:
            return "fmax(val_t%s_c%s - ibw_t%s_c%s/2.0, %s)" % (a, b, a, b, self.lower_bound(cols))
        else:
            return "val_t%s_c%s - ibw_t%s_c%s/2.0" % (a, b, a, b)

    def generateEstimateCode(self,f, query,base,node,stats):
        icols = Utils.generateInvariantColumns(query)
        for j, join in enumerate(query.joins):
            print >> f, "%sT iu_j%s = %s;" % ("    " * (base), j, self.upper_bound(join[:]))
            print >> f, "%sT il_j%s = %s;" % ("    " * (base), j, self.lower_bound(join[:]))
            print >> f, "%sT ou_j%s = iu_j%s + 1;" % ("    " * (base), j, j)
            print >> f, "%sT ol_j%s = il_j%s - 1;" % ("    " * (base), j, j)
            print >> f, "%sT j%s = 0.0;" % ("    " * (base), j)

            print >> f, "%sif(iu_j%s - il_j%s >= 0.0 && ou_j%s - ol_j%s > 0.0){" % ("    " * (base), j, j, j, j)
            base += 1
            print >> f, "%sj%s += (iu_j%s - il_j%s);" % ("    " * (base), j, j, j)
            print >> f, "%sj%s += %s;" % ("    " * (base), j, '*'.join(
                map(lambda x: "(il_j%s -fmax(val_t%s_c%s-bw_t%s_c%s/2.0,ol_j%s))" % (j, x[0], x[1], x[0], x[1], j), join)))
            print >> f, "%sj%s += %s;" % ("    " * (base), j, '*'.join(
                map(lambda x: "(min(val_t%s_c%s+bw_t%s_c%s/2.0,ou_j%s)-iu_j%s)" % (x[0], x[1], x[0], x[1], j, j), join)))
            base -= 1
            print >> f, "%s} else if(iu_j%s - il_j%s < 0.0 && ou_j%s - ol_j%s > 0.0) {" % ("    " * (base), j, j, j, j)
            base += 1
            print >> f, "%sj%s = %s;" % ("    " * (base), j, '*'.join(map(
                lambda x: "(min(val_t%s_c%s+bw_t%s_c%s/2.0,ou_j%s)-fmax(val_t%s_c%s-bw_t%s_c%s/2.0,ol_j%s))" % (
                x[0], x[1], x[0], x[1], j, x[0], x[1], x[0], x[1], j), join)))

            base -= 1
            print >> f, "%s}" % ("    " * (base))
            print >> f, "%sj%s /= %s;" % (
            ("    " * (base), j, '*'.join(map(lambda x: "bw_t%s_c%s" % (x[0], x[1]), join))))

        print >> f, "%ssum += 1.0 " % ("    " * base),
        for j, join in enumerate(query.joins):
            print >> f, "* j%s" % j,
        for i, pt in enumerate(join):
            a, b = pt
            if a == 0 and len(icols[a]) > 0:
                print >> f, "* c_t%s" % (a),
            else:
                if len(icols[a]) > 0:
                    print >> f, "* c_t%s" % (a),
        print >> f, ";"

    def generateCountCode(self,f, query,base,node,stats):
        print >> f, "%ssum += 1.0;" % ("    " * base),

    def generatePreamble(self,f,query):
        print >>f, "    T sum = 0.0;"
        #Create all
        for x,join in enumerate(query.joins):
            for pt in join:
                a,b = pt
                print >>f, "    bw_t%s_c%s = fmax(1.0,bw_t%s_c%s * 3.07634);" % (a,b,a,b)
                print >>f, "    unsigned int ibw_t%s_c%s = 2* ((unsigned int) (bw_t%s_c%s+1.0)/2) - 1;" % (a,b,a,b)
                print >>f, "    unsigned int obw_t%s_c%s = ibw_t%s_c%s + 2;" % (a,b,a,b)
        print >>f


def generateJoinEstimateKernel(f,query,estimator,stats):
    print >>f, "__kernel void estimate("
    icols = Utils.generateInvariantColumns(query)   
    jcols = Utils.generateJoinColumns(query)

    _ , dvals = stats
    graph = constructJoinGraph(query)
    t1, c1 = graph.left_col
    t2, c2 = graph.right_col

    tids = graph.collectTableIDs()
    pairs = graph.collectJoinPairs()

    if estimator.join_kernel == "Cont":
        kde = ContKDEKernel()
    elif estimator.join_kernel == "Rect":
        kde = RectKDEKernel()
    elif estimator.join_kernel == "Cat":
        kde = CatKDEKernel()

    for x,t in enumerate(tids):
        for jc in jcols[t]:
            print >>f, "    __global unsigned int* t%s_c%s," % (t,jc),
            print >>f, "    double bw_t%s_c%s," % (t,jc)
        if icols[t]:
            print >>f, "    __global double* inv_t%s," % (t)
        if x > 0:
            print >>f, "    unsigned int n_t%s," % (t)
    #Here we go.
    for t1,c1,t2,c2 in pairs:
        print >> f, "    double limit_t%s_c%s_t%s_c%s," % (t1,c1,t2,c2)
    if estimator.join_kernel == "Cat":
        print >> f, "    double omega,"
    print >>f, "    __global double* contributions, unsigned int ss){"

    print >> f
    #We start of with table 1.
    kde.generatePreamble(f,query)

    print >>f, "     for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){"
    print >>f, "        if (offset + get_global_id(0) < ss){"

    graph.generateJoinEstimateKernelBottomUp(f, query, estimator)
    kde.generateEstimateCode(f,query,graph.jid+1,graph,stats)
    graph.generateJoinEstimateKernelTopDown(f, query)

    if estimator.join_kernel == "Cat":
        print >> f, "   T jnone = (1.0-bw_t%s_c%s) * bw_t%s_c%s / (%f-1.0) + (1.0-bw_t%s_c%s) * bw_t%s_c%s / (%f-1.0) + bw_t%s_c%s*bw_t%s_c%s * (%f-2.0) / ((%f-1.0)*(%f-1.0));" % (t1,c1,t2,c2,dvals[t2][c2],
                                                                                                                                                                                                t2,c2,t1,c1,dvals[t1][c1],
                                                                                                                                                                                                t1,c1,t2,c2,min(dvals[t1][c1],dvals[t2][c2]),dvals[t1][c1],dvals[t2][c2])
        t1, c1 = graph.left_col
        print >>f, "     sum += c_t%s * jnone * (omega-osum);" % (t1)
    print >>f, "    }"
    print >>f, "    }"

    print >>f, "    if (get_global_id(0) < ss) contributions[get_global_id(0)] = sum;"
    print >>f, "}"

#Classes representing a left-deep join tree
    
def generateCIncludes(f):
    print >>f, """
    
#include <iostream>
#include <string>
#include <streambuf>
#include <nlopt.h>
#include <sstream>
#include <cmath>
#include <fstream>
#include <cstdio>

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

std::chrono::time_point<std::chrono::high_resolution_clock> opt_start;
"""
    
def generateGPUJKDEGlobalTraining(cf,query,estimator):
    icols = Utils.generateInvariantColumns(query)                  

    #Generate contribution arrays              
    print >>cf, "    std::string training_cardinality_string = iteration_stream.str() + \"/training_join_true.dump\";"
    print >>cf, "    p.j_training_cardinality = readUArrayFromFile(training_cardinality_string.c_str());"    
    for i,indices in enumerate(icols):
        if len(indices) != 0:
            for j in indices:
                if estimator.kernels[i][j] == "GaussRange":
                    print >>cf, "    std::string training_j_l_t%s_c%s_string = iteration_stream.str() + \"/training_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                    print >>cf, "    p.j_training_l_t%s_c%s= readUArrayFromFile(training_j_l_t%s_c%s_string.c_str());" % (i,j,i,j)
                    print >>cf, "    std::string training_j_u_t%s_c%s_string = iteration_stream.str() + \"/training_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                    print >>cf, "    p.j_training_u_t%s_c%s = readUArrayFromFile(training_j_u_t%s_c%s_string.c_str());" % (i,j,i,j)
                else:
                    print >>cf, "    std::string training_j_p_t%s_c%s_string = iteration_stream.str() + \"/training_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                    print >>cf, "    p.j_training_p_t%s_c%s = readUArrayFromFile(training_j_p_t%s_c%s_string.c_str());" % (i,j,i,j)  
            print >>cf
    
    print >>cf, "    double* bw = (double*) calloc(%s,sizeof(double));" % (len(Utils.flatten(estimator.kernels)))        
    print >>cf, "    double ub[%s] = {0.0};" % (len(Utils.flatten(estimator.kernels))) 
    print >>cf, "    double lb[%s] = {0.0};" % (len(Utils.flatten(estimator.kernels))) 
    
    i = 0
    for x,kernels in enumerate(estimator.kernels):
        for y,kernel in enumerate(kernels):
            if kernel == "GaussRange" or kernel == "GaussPoint":
                print >>cf, "    ub[%s] = fmax(p.bw_t%s_c%s,2.0);" % (i,x,y)
                print >>cf, "    lb[%s] = 0.1;" % (i) 
                print >>cf, "    bw[%s] = 2.0;" % (i)
            elif kernel == "CategoricalPoint":
                print >>cf, "    lb[%s] = DBL_EPSILON;" % (i) 
                print >>cf, "    bw[%s] = 1.0/p.ss%s;" % (i,x) 
                print >>cf, "    ub[%s] = 1.0-DBL_EPSILON;" % (i)
            else:
                print (y,kernel)
                raise Exception("Wuut wuut?")
            i += 1

    print >> cf, "    double minf = 0.0;"    
    #print >> cf, "    std::string bwstr(argv[0]);"
    #print >> cf, "    bwstr.append(\".bw_dump\");"
    #print >> cf, "    if(fexists(bwstr.c_str())) bw = readDArrayFromFile(bwstr.c_str());"

    i = 0
    for x,kernels in enumerate(estimator.kernels):
        for y,kernel in enumerate(kernels):
            print >>cf, "    ub[%s] = fmax(bw[%s],ub[%s]);" % (i,i,i)
            i += 1

    #The categorical kernel needs global optimization urgently
    if estimator.join_kernel == "Cat":
        print >>cf, """
        nlopt_opt gopt = nlopt_create(NLOPT_GN_MLSL,%s);
        nlopt_set_lower_bounds(gopt,lb);
        nlopt_set_upper_bounds(gopt,ub);
        nlopt_set_min_objective(gopt,obj,&p);
    """ % (len(Utils.flatten(estimator.kernels)))
        print >>cf

        print >>cf, """
        nlopt_set_maxeval(gopt, %s);
        nlopt_set_ftol_rel(gopt, %s);
        nlopt_opt lopt = nlopt_create(NLOPT_LN_COBYLA,%s);
        nlopt_set_lower_bounds(lopt,lb);
        nlopt_set_upper_bounds(lopt,ub);
        nlopt_set_local_optimizer(gopt, lopt);
        int grc = nlopt_optimize(gopt, bw, &minf);
        assert(grc >=0);
    """ % (40,"1e-10",len(Utils.flatten(estimator.kernels)))
    print >>cf, "   opt_start = std::chrono::high_resolution_clock::now();"
    print >> cf, """
            nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA,%s);
            nlopt_set_lower_bounds(opt,lb);
            nlopt_set_upper_bounds(opt,ub);
            nlopt_set_maxeval(opt, %s);
            nlopt_set_ftol_rel(opt, %s);
            nlopt_set_min_objective(opt,obj,&p);
            p.opt = &opt;
            int frc = nlopt_optimize(opt, bw, &minf);
            assert(frc >=0);
        """ % (len(Utils.flatten(estimator.kernels)), 1000, "1e-5")
    
    #print >> cf, "    ddump(bw, %s, bwstr.c_str());" % len(Utils.flatten(estimator.kernels))

    i=0
    for x,kernels in enumerate(estimator.kernels):
        for y,kernel in enumerate(kernels):
            print >>cf, "    p.bw_t%s_c%s = bw[%s];" % (x,y,i)
            i += 1
    print >>cf

    
def generateGPUJKDECode(i,query,estimator,stats,cu_factor):
    ts, dv = stats
    graph = constructJoinGraph(query)
    tids = graph.collectTableIDs()

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
        
        print >>cf, "//"
        graph.generateTableEstimateKernel(cf,query,estimator,stats)
        generateBinarySearchCode(cf)
        generateJoinEstimateKernel(cf,query,estimator,stats)
        print >>cf, "//"
                
        #Do we need table level estimation kernels?
        if estimator.bw_optimization == "local":
            for j, kernels in enumerate(estimator.kernels):
                generateTableGradientContributionKernel(cf,"grad_t%s" % j,kernels,dv[j])
        
    with open("./%s_AGPUJKDE.cpp" % i,'w') as cf:
        generateCIncludes(cf)
        generateRectKDELimit(cf)
        generateContKDELimit(cf)
        generateGPUJKDEParameterArray(cf,query,estimator)
        Utils.generateGPUVectorConverterFunction(cf)
        Utils.generateUintFileReaderFunction(cf)
        Utils.generateDoubleFileReaderFunction(cf)
        Utils.generateFileCheckFunction(cf)
        Utils.generateScottBWFunction(cf)
        Utils.generateDoubleDumper(cf)
        generateGPUJKDEEstimateFunction(cf,graph,query,estimator,prod(ts.values())**-1.0,stats,cu_factor)
        generateGPUJKDETestWrapper(cf,query,estimator)
        
        if estimator.bw_optimization == "local":
            for tid,table in enumerate(query.tables):
                generateTableEstGrad(cf,tid,query,estimator)
                generateTableObjectiveGrad(cf,tid,query,estimator)
        elif estimator.bw_optimization == "join":
                generateGPUJKDEObjective(cf,query,estimator)
        
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
                if estimator.kernels[j][k] == "GaussPoint" or estimator.kernels[j][k] == "GaussRange":
                    print >>cf, "    p.bw_t%s_c%s = scott_bw(s_t%s_c%s,  p.ss%s, %s);" % (j,k,j,k,j,len(query.tables))
                    print >>cf, "    if(p.bw_t%s_c%s < 0.2) p.bw_t%s_c%s = 0.2;" % (j,k,j,k)
                else:
                    print >>cf, "    p.bw_t%s_c%s = 1.0/(1.0+1.0/%f);" % (j,k,dv[j][k]-1)
            print >>cf

        for t,cs in enumerate(jcols):
            if cols[t]:
                for c in cs:
                    print >> cf, "    p.sr_t%s_c%s = compute::vector<unsigned int>(p.ss%s, p.ctx);" % (t,c,t)
        print >> cf, "    p.final_contributions = compute::vector<double>(p.ss%s, p.ctx);" % (tids[0])
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


            #Prepare training
        if estimator.bw_optimization == "local":
            generateGPUJKDELocalTraining(cf,query,estimator,cu_factor)
        elif estimator.bw_optimization == "join":
            if estimator.join_kernel == "Rect":
                raise Exception("This is not how optimization on join works.")
            elif estimator.join_kernel == "Cont":
                generateGPUJKDEGlobalTraining(cf,query,estimator)
            elif estimator.join_kernel == "Cat":
                generateGPUJKDEGlobalTraining(cf,query,estimator)
            else:
                raise Exception("I don't know this join kernel.")
        else:
            raise Exception("I don't know this type of join optimization.")
            
        print >>cf, "    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";"
        print >>cf, "    p.j_test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());"
        
        for i,indices in enumerate(cols):
            if len(indices) != 0:
                for j in indices:
                    if estimator.kernels[i][j] == "GaussRange":
                        print >>cf, "    std::string j_l_t%s_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                        print >>cf, "    p.j_l_t%s_c%s= readUArrayFromFile(j_l_t%s_c%s_string.c_str());" % (i,j,i,j)
                        print >>cf, "    std::string j_u_t%s_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                        print >>cf, "    p.j_u_t%s_c%s = readUArrayFromFile(j_u_t%s_c%s_string.c_str());" % (i,j,i,j)
                    else:
                        print >>cf, "    std::string j_p_t%s_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid)
                        print >>cf, "    p.j_p_t%s_c%s = readUArrayFromFile(j_p_t%s_c%s_string.c_str());" % (i,j,i,j)
        
        print >>cf
        print >>cf, "    join_test(&p);"
        print >>cf, "}"   
        

def generateGPUJKDEObjective(f,query,estimator):
    icols = Utils.generateInvariantColumns(query)
    print >>f, "double obj(unsigned n, const double* bw, double* grad, void* f_data){"

    print >>f, "    parameters* p = (parameters*) f_data;"

    i = 0
    for x,table in enumerate(query.tables):
        for y,col in enumerate(table.columns):
            print >>f, "    p->bw_t%s_c%s = bw[%s];" % (x,y,i)
            i += 1

    print >>f, "    int first = 1;"
    print >>f, "    double est = 0.0;"
    if estimator.objective == "squared":
        print >>f, "    double objective = 0.0;"
    elif estimator.objective == "Q":
        print >>f, "    double objective = 1.0;"
    else:
        raise Exception("I don't know this objective function.")

    print >>f, "    for(unsigned int i = 0; i < %s; i++){" % estimator.training
    if hasattr(estimator, 'limit_opt'):
        print >>f, "    if(std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now()-opt_start).count() > %s)" % estimator.limit_opt
        print >>f, "        nlopt_force_stop(*(p->opt));"
    print >>f, "        if(first ",
    for x, cols in enumerate(icols):
        for y in cols:
            if estimator.kernels[x][y] == "GaussRange":
                print >>f, "|| p->j_training_u_t%s_c%s[i] !=  p->j_training_u_t%s_c%s[i-1]" % (x,y,x,y)
                print >>f, "|| p->j_training_l_t%s_c%s[i] !=  p->j_training_l_t%s_c%s[i-1]" % (x,y,x,y),
            else:
                print >>f, "|| p->j_training_p_t%s_c%s[i] != p->j_training_p_t%s_c%s[i-1]" % (x,y,x,y),
    print >> f, "){"
    print >>f, "            first = 0;"
    print >>f, "            est =  join_estimate_instance(p ",
    for x, cols in enumerate(icols):
        for y in cols: 
            if estimator.kernels[x][y] == "GaussRange":
                print >>f, ", p->j_training_u_t%s_c%s[i], p->j_training_l_t%s_c%s[i] " % (x,y,x,y),
            else:
                print >>f, ", p->j_training_p_t%s_c%s[i] " % (x,y),
    print >>f, ");"
    print >> f, "        }"
    print >>f, "        unsigned int trues =  p->j_training_cardinality[i];"

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
        
        
#Generate parameter struct that is passed to the estimation/gradient functions
def generateGPUJKDEParameterArray(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    jcols = Utils.generateJoinColumns(query)
    print >>f, """
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
"""
    print >>f, "    unsigned int iteration;"
    print >>f, "    nlopt_opt* opt;"
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
    print >>f, "    compute::vector<double> final_contributions;"
    #Training
    if estimator.bw_optimization == "local":
        for tid,tab in enumerate(query.tables):
            print >>f, "    size_t global_t%s;" % (tid)
            print >>f, "    size_t local_t%s;" % (tid)
            print >>f, "    compute::kernel estk%s;" % (tid)
            print >>f, "    compute::kernel gradk%s;" % (tid)
            print >>f, "    compute::vector<double> est_t%s;" % (tid)
            print >>f, "    unsigned int* true_t%s;" % tid 
            for cid,col in enumerate(tab.columns):
                print >>f, "    compute::vector<double> grad_t%s_c%s;" % (tid,cid)
                if estimator.kernels[tid][cid] == "GaussRange":
                    print >>f, "    unsigned int* l_t%s_c%s;" % (tid,cid)
                    print >>f, "    unsigned int* u_t%s_c%s;" % (tid,cid)
                else:
                    print >>f, "    unsigned int* p_t%s_c%s;" % (tid,cid)

    print >>f    
    print >>f, "    unsigned int* j_test_cardinality;"
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        if len(indices) != 0:
            for j in indices:
                if estimator.kernels[i][j] == "GaussRange":
                    print >>f, "    unsigned int* j_l_t%s_c%s;" % (i,j)
                    print >>f, "    unsigned int* j_u_t%s_c%s;" % (i,j)
                else:
                    print >>f, "    unsigned int* j_p_t%s_c%s;" % (i,j)
    if estimator.bw_optimization == "join":
        print >>f, "    unsigned int* j_training_cardinality;"
                
        for i,indices in enumerate(cols):
        #Start with computing the invariant contributions   
            if len(indices) != 0:
                for j in indices:
                    if estimator.kernels[i][j] == "GaussRange":
                        print >>f, "    unsigned int* j_training_l_t%s_c%s;" % (i,j)
                        print >>f, "    unsigned int* j_training_u_t%s_c%s;" % (i,j)
                    else:
                        print >>f, "    unsigned int* j_training_p_t%s_c%s;" % (i,j)
                
    print >>f, """
} parameters;
"""


def generateGPUJKDEEstimateFunction(f, nodes, query, estimator, limit, stats, cu_factor):
     icols = Utils.generateInvariantColumns(query)
     jcols = Utils.generateJoinColumns(query)
     ts, dv = stats

     print >> f, "double join_estimate_instance(parameters* p"
     for i, indices in enumerate(icols):
         # Start with computing the invariant contributions
         if len(indices) != 0:
             for j in indices:
                 if estimator.kernels[i][j] == "GaussRange":
                     print >> f, "    , unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (i, j, i, j)
                 else:
                     print >> f, "    , unsigned int p_t%s_c%s" % (i, j)
     print >> f
     print >> f, "){"

     nodes.generateTableCode(f, query, estimator, limit, cu_factor)

     if estimator.join_kernel == "Cat":
         if len(query.joins) > 1 or len(query.joins[0]) != 2:
             raise Exception("The categorical kernel does not support more than two joins.")

         # Compute omega_2
         t2, _ = nodes.right_col
         print >> f, "    double omega_2 = 1.0;"
         if icols[t2]:
             print >> f, "    boost::compute::reduce(p->invr_t%s.begin(),p->invr_t%s.begin()+rss_t%s, &omega_2, p->queue);" % (
             t2, t2, t2)
             print >> f, "    p->queue.finish();"

     # Next, generate the limits
     pairs = nodes.collectJoinPairs()
     tids = nodes.collectTableIDs()

     for t1, c1, t2, c2 in pairs:
         if estimator.join_kernel == "Cont":
             print >> f, "    double limit_t%s_c%s_t%s_c%s =  compute_cont_limit(p->bw_t%s_c%s, p->bw_t%s_c%s, p->ts%s, p->ts%s);" % (
             t1, c1, t2, c2, t1, c1, t2, c2, t1, t2)
         elif estimator.join_kernel == "Rect":
             print >> f, "    double limit_t%s_c%s_t%s_c%s =  compute_rect_limit(p->bw_t%s_c%s, p->bw_t%s_c%s);" % (
             t1, c1, t2, c2, t1, c1, t2, c2)
         elif estimator.join_kernel == "Cat":
             print >> f, "    double limit_t%s_c%s_t%s_c%s =  0.0;" % (t1, c1, t2, c2)
         else:
             raise Exception("Unsupported join kernel.")

     print >> f, "    size_t local = 64;"
     print >> f, "    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((rss_t%s-1)/local+1)*local);" % (cu_factor,tids[0])

     print >> f, "    p->estimate.set_args(",

     for x, t in enumerate(tids):
         for jc in jcols[t]:
             if icols[t]:
                 print >> f, "    p->sr_t%s_c%s," % (t, jc),
             else:
                 print >> f, "    p->s_t%s_c%s," % (t, jc),
             print >> f, "    p->bw_t%s_c%s," % (t, jc)
         if icols[t]:
             print >> f, "    p->invr_t%s," % (t)
         if x > 0:
             print >> f, "    (unsigned int) rss_t%s," % (t)
     # Here we go.
     for t1, c1, t2, c2 in pairs:
         print >> f, "    limit_t%s_c%s_t%s_c%s," % (t1, c1, t2, c2)
     if estimator.join_kernel == "Cat":
         print >> f, "    omega_2,"
     print >> f, "    p->final_contributions, (unsigned int) rss_t%s);" % tids[0]
     print >> f, "    p->queue.enqueue_nd_range_kernel(p->estimate,1,NULL,&(global), &(local));"

     print >> f, "    double est = 0.0;"
     print >> f, "    boost::compute::reduce(p->final_contributions.begin(), p->final_contributions.begin()+std::min(global,rss_t%s), &est, p->queue);" % (
     tids[0])
     print >> f, "    p->queue.finish();"
     for i, _ in enumerate(query.tables):
         print >> f, "    est *= ((double) p->ts%s)/p->ss%s;" % (i, i)
     print >> f, "    return est;"
     print >> f, "}"
