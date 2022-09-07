 #Code generator for gauss kernel
import Utils
from functools import reduce
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
    print("""
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
    """, file=f)
    
def generateRectKDELimit(f):
    print("""
    double compute_rect_limit(double bw1, double bw2){
        bw1 = fmax(1.0,bw1 * 3.07634);
        bw2 = fmax(1.0,bw2 * 3.07634);
        double obw1 = 2* ((unsigned int) (bw1+1.0)/2)+1;
        double obw2 = 2* ((unsigned int) (bw2+1.0)/2)+1;
        return (obw1+obw2)/2;
    }
    """, file=f)

def generateContKDELimit(f):
    print("""
    double compute_cont_limit(double bw1, double bw2, double t1, double t2){
        double bw_term = bw1*bw1 + bw2*bw2;
        return  sqrt(-2.0 * bw_term * log(sqrt(2*M_PI*bw_term)/ (t1*t2)));
    }
    """, file=f)

class CatKDEKernel:
    def generateEstimateCode(self,f, query,base,node,stats):
        ts,dvals = stats
        t1,c1 = node.left_col
        t2, c2 = node.right_col

        icols = Utils.generateInvariantColumns(query)
        print("%sT j = (1-bw_t%s_c%s)*(1-bw_t%s_c%s) +  bw_t%s_c%s * bw_t%s_c%s / (%f-1.0);" % ("    " * base,t1,c1,t2,c2,t1,c1,t2,c2,max(dvals[t1][c1],dvals[t2][c2])), file=f)

        print("%ssum += j " % ("    " * base), end=' ', file=f)
        if icols[t1]:
            print("* c_t%s" % (t1), end=' ', file=f)
        if icols[t2]:
            print("* c_t%s" % (t2), end=' ', file=f)
        print(";", file=f)
        if icols[t2]:
            print("%sosum += c_t%s;" % ("    " * base,t2), file=f)

    def generateCountCode(self,f, query,base,node,stats):
        print("%ssum += 1.0;" % ("    " * base), file=f)


    def generatePreamble(self,f,query):
        print("    T sum = 0.0;", file=f)
        print("    T osum = 0.0;", file=f)

   
class ContKDEKernel:
    def generateEstimateCode(self,f, query,base,node,stats):
        if len(query.joins) == 1 and len(query.joins[0]) == 2:
            self.generate2EstimateCode(f, query,base,node,stats)
            return
            
        icols = Utils.generateInvariantColumns(query)
        for j, join in enumerate(query.joins):
            print("%sunsigned int min_val_j%s = 0;" % ("    " * (base), j), file=f)
            for pt in join:
                a, b = pt
                print("%smin_val_j%s = min(min_val_j%s,val_t%s_c%s);" % ("    " * (base), j, j, a, b), file=f)
            for pt in join:
                a, b = pt
                print("%sunsigned int sval_t%s_c%s = val_t%s_c%s-min_val_j%s;" % ("    " * (base),a,b,a,b,j), file=f)

            print("%sT mu_sum_j%s = (0.0 " % ("    " * (base), j), end=' ', file=f)
            for pt in join:
                a, b = pt
                print("+ sval_t%s_c%s / bw_t%s_c%s" % (a, b, a, b), end=' ', file=f)
            print(") / sigma_sum_inv_j%s;" % j, file=f)

            print("%sT scaled_mu_sum_j%s = 0.0" % ("    " * base, j), end=' ', file=f)
            for pt in join:
                a, b = pt
                print("+ ((T)sval_t%s_c%s)*sval_t%s_c%s / bw_t%s_c%s" % (a, b, a, b, a, b), end=' ', file=f)
            print(";", file=f)

            print("%sT j%s = exp(-0.5 *(scaled_mu_sum_j%s - mu_sum_j%s * mu_sum_j%s * sigma_sum_inv_j%s));" % (
            "    " * base, j, j, j, j, j), file=f)

        print("%ssum += 1.0 " % ("    " * base), end=' ', file=f)
        for j, join in enumerate(query.joins):
            print("* j%s * factor_j%s * pow(0.5*M_1_PI,%s)" % (j,j,(len(join)-1)/2.0), end=' ', file=f)
        for i,c in enumerate(icols):
            if len(c) != 0:
                print("* c_t%s" % (i), end=' ', file=f)
        print(";", file=f)

    def generate2EstimateCode(self,f, query,base,node,stats):
        icols = Utils.generateJoinColumns(query)
        if len(query.joins) != 1:
            raise Exception("This feature was disabled due to huge ass numerical instabilities.")

        join = query.joins[0]
        t1,c1 = join[0]
        t2,c2 = join[1]

        print("%ssum += M_SQRT1_2 * M_2_SQRTPI * 0.5 / resbw * exp( -0.5 * (((T)val_t%s_c%s)-((T)val_t%s_c%s))*(((T)val_t%s_c%s)-((T)val_t%s_c%s))/(resbw*resbw)) " % ("    " * base,t1,c1,t2,c2,t1,c1,t2,c2), end=' ', file=f)
        for i, pt in enumerate(join):
            a, b = pt
            print("* c_t%s" % (a), end=' ', file=f)
        print(";", file=f)

    def generateCountCode(self,f, query,base,node,stats):
        print("%ssum += 1.0;" % ("    " * base), end=' ', file=f)


    def generatePreamble(self,f,query):
        if len(query.joins) == 1 and len(query.joins[0]) == 2:
            self.generate2Preamble(f, query)
            return

        print("    T sum = 0.0;", file=f)
        # Create all
        for x, join in enumerate(query.joins):
            for pt in join:
                a, b = pt
                print("    bw_t%s_c%s *= bw_t%s_c%s;" % (a, b, a, b), file=f)
        print(file=f)

        for x, join in enumerate(query.joins):
            print("    T sigma_prod_j%s = 1.0" % x, end=' ', file=f)
            for pt in join:
                a, b = pt
                print("* bw_t%s_c%s" % (a, b), end=' ', file=f)
            print(";", file=f)

        for x, join in enumerate(query.joins):
            print("    T sigma_sum_inv_j%s = 0.0" % x, end=' ', file=f)
            for pt in join:
                a, b = pt
                print("+ 1.0/(bw_t%s_c%s)" % (a, b), end=' ', file=f)
            print(";", file=f)

            print(file=f)
            print("    T factor_j%s = sqrt(1.0/(sigma_prod_j%s*sigma_sum_inv_j%s));" % (x, x, x), file=f)


    def generate2Preamble(self,f,query):
        print("    T sum = 0.0;", file=f)
        print("    T resbw = sqrt(0.0 + ", file=f)
        # Create all
        for x, join in enumerate(query.joins):
            for pt in join:
                a, b = pt
                print(" + bw_t%s_c%s * bw_t%s_c%s" % (a, b, a, b), end=' ', file=f)
        print(");", file=f)
        print(file=f)

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
            print("%sT iu_j%s = %s;" % ("    " * (base), j, self.upper_bound(join[:])), file=f)
            print("%sT il_j%s = %s;" % ("    " * (base), j, self.lower_bound(join[:])), file=f)
            print("%sT ou_j%s = iu_j%s + 1;" % ("    " * (base), j, j), file=f)
            print("%sT ol_j%s = il_j%s - 1;" % ("    " * (base), j, j), file=f)
            print("%sT j%s = 0.0;" % ("    " * (base), j), file=f)

            print("%sif(iu_j%s - il_j%s >= 0.0 && ou_j%s - ol_j%s > 0.0){" % ("    " * (base), j, j, j, j), file=f)
            base += 1
            print("%sj%s += (iu_j%s - il_j%s);" % ("    " * (base), j, j, j), file=f)
            print("%sj%s += %s;" % ("    " * (base), j, '*'.join(
                ["(il_j%s -fmax(val_t%s_c%s-bw_t%s_c%s/2.0,ol_j%s))" % (j, x[0], x[1], x[0], x[1], j) for x in join])), file=f)
            print("%sj%s += %s;" % ("    " * (base), j, '*'.join(
                ["(min(val_t%s_c%s+bw_t%s_c%s/2.0,ou_j%s)-iu_j%s)" % (x[0], x[1], x[0], x[1], j, j) for x in join])), file=f)
            base -= 1
            print("%s} else if(iu_j%s - il_j%s < 0.0 && ou_j%s - ol_j%s > 0.0) {" % ("    " * (base), j, j, j, j), file=f)
            base += 1
            print("%sj%s = %s;" % ("    " * (base), j, '*'.join(["(min(val_t%s_c%s+bw_t%s_c%s/2.0,ou_j%s)-fmax(val_t%s_c%s-bw_t%s_c%s/2.0,ol_j%s))" % (
                x[0], x[1], x[0], x[1], j, x[0], x[1], x[0], x[1], j) for x in join])), file=f)

            base -= 1
            print("%s}" % ("    " * (base)), file=f)
            print("%sj%s /= %s;" % (
            ("    " * (base), j, '*'.join(["bw_t%s_c%s" % (x[0], x[1]) for x in join]))), file=f)

        print("%ssum += 1.0 " % ("    " * base), end=' ', file=f)
        for j, join in enumerate(query.joins):
            print("* j%s" % j, end=' ', file=f)
        for i, pt in enumerate(join):
            a, b = pt
            if a == 0 and len(icols[a]) > 0:
                print("* c_t%s" % (a), end=' ', file=f)
            else:
                if len(icols[a]) > 0:
                    print("* c_t%s" % (a), end=' ', file=f)
        print(";", file=f)

    def generateCountCode(self,f, query,base,node,stats):
        print("%ssum += 1.0;" % ("    " * base), end=' ', file=f)

    def generatePreamble(self,f,query):
        print("    T sum = 0.0;", file=f)
        #Create all
        for x,join in enumerate(query.joins):
            for pt in join:
                a,b = pt
                print("    bw_t%s_c%s = fmax(1.0,bw_t%s_c%s * 3.07634);" % (a,b,a,b), file=f)
                print("    unsigned int ibw_t%s_c%s = 2* ((unsigned int) (bw_t%s_c%s+1.0)/2) - 1;" % (a,b,a,b), file=f)
                print("    unsigned int obw_t%s_c%s = ibw_t%s_c%s + 2;" % (a,b,a,b), file=f)
        print(file=f)


def generateJoinEstimateKernel(f,query,estimator,stats):
    print("__kernel void estimate(", file=f)
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
            print("    __global unsigned int* t%s_c%s," % (t,jc), end=' ', file=f)
            print("    double bw_t%s_c%s," % (t,jc), file=f)
        if icols[t]:
            print("    __global double* inv_t%s," % (t), file=f)
        if x > 0:
            print("    unsigned int n_t%s," % (t), file=f)
    #Here we go.
    for t1,c1,t2,c2 in pairs:
        print("    double limit_t%s_c%s_t%s_c%s," % (t1,c1,t2,c2), file=f)
    if estimator.join_kernel == "Cat":
        print("    double omega,", file=f)
    print("    __global double* contributions, unsigned int ss){", file=f)

    print(file=f)
    #We start of with table 1.
    kde.generatePreamble(f,query)

    print("     for(unsigned int offset = 0; offset < ss; offset += get_global_size(0)){", file=f)
    print("        if (offset + get_global_id(0) < ss){", file=f)

    graph.generateJoinEstimateKernelBottomUp(f, query, estimator)
    kde.generateEstimateCode(f,query,graph.jid+1,graph,stats)
    graph.generateJoinEstimateKernelTopDown(f, query)

    if estimator.join_kernel == "Cat":
        print("   T jnone = (1.0-bw_t%s_c%s) * bw_t%s_c%s / (%f-1.0) + (1.0-bw_t%s_c%s) * bw_t%s_c%s / (%f-1.0) + bw_t%s_c%s*bw_t%s_c%s * (%f-2.0) / ((%f-1.0)*(%f-1.0));" % (t1,c1,t2,c2,dvals[t2][c2],
                                                                                                                                                                                                t2,c2,t1,c1,dvals[t1][c1],
                                                                                                                                                                                                t1,c1,t2,c2,min(dvals[t1][c1],dvals[t2][c2]),dvals[t1][c1],dvals[t2][c2]), file=f)
        t1, c1 = graph.left_col
        print("     sum += c_t%s * jnone * (omega-osum);" % (t1), file=f)
    print("    }", file=f)
    print("    }", file=f)

    print("    if (get_global_id(0) < ss) contributions[get_global_id(0)] = sum;", file=f)
    print("}", file=f)

#Classes representing a left-deep join tree
    
def generateCIncludes(f):
    print("""
    
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
""", file=f)
    
def generateGPUJKDEGlobalTraining(cf,query,estimator):
    icols = Utils.generateInvariantColumns(query)                  

    #Generate contribution arrays              
    print("    std::string training_cardinality_string = iteration_stream.str() + \"/training_join_true.dump\";", file=cf)
    print("    p.j_training_cardinality = readUArrayFromFile(training_cardinality_string.c_str());", file=cf)    
    for i,indices in enumerate(icols):
        if len(indices) != 0:
            for j in indices:
                if estimator.kernels[i][j] == "GaussRange":
                    print("    std::string training_j_l_t%s_c%s_string = iteration_stream.str() + \"/training_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                    print("    p.j_training_l_t%s_c%s= readUArrayFromFile(training_j_l_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                    print("    std::string training_j_u_t%s_c%s_string = iteration_stream.str() + \"/training_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                    print("    p.j_training_u_t%s_c%s = readUArrayFromFile(training_j_u_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                else:
                    print("    std::string training_j_p_t%s_c%s_string = iteration_stream.str() + \"/training_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                    print("    p.j_training_p_t%s_c%s = readUArrayFromFile(training_j_p_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)  
            print(file=cf)
    
    print("    double* bw = (double*) calloc(%s,sizeof(double));" % (len(Utils.flatten(estimator.kernels))), file=cf)        
    print("    double ub[%s] = {0.0};" % (len(Utils.flatten(estimator.kernels))), file=cf) 
    print("    double lb[%s] = {0.0};" % (len(Utils.flatten(estimator.kernels))), file=cf) 
    
    i = 0
    for x,kernels in enumerate(estimator.kernels):
        for y,kernel in enumerate(kernels):
            if kernel == "GaussRange" or kernel == "GaussPoint":
                print("    ub[%s] = fmax(p.bw_t%s_c%s,2.0);" % (i,x,y), file=cf)
                print("    lb[%s] = 0.1;" % (i), file=cf) 
                print("    bw[%s] = 2.0;" % (i), file=cf)
            elif kernel == "CategoricalPoint":
                print("    lb[%s] = DBL_EPSILON;" % (i), file=cf) 
                print("    bw[%s] = 1.0/p.ss%s;" % (i,x), file=cf) 
                print("    ub[%s] = 1.0-DBL_EPSILON;" % (i), file=cf)
            else:
                print((y,kernel))
                raise Exception("Wuut wuut?")
            i += 1

    print("    double minf = 0.0;", file=cf)    
    #print >> cf, "    std::string bwstr(argv[0]);"
    #print >> cf, "    bwstr.append(\".bw_dump\");"
    #print >> cf, "    if(fexists(bwstr.c_str())) bw = readDArrayFromFile(bwstr.c_str());"

    i = 0
    for x,kernels in enumerate(estimator.kernels):
        for y,kernel in enumerate(kernels):
            print("    ub[%s] = fmax(bw[%s],ub[%s]);" % (i,i,i), file=cf)
            i += 1

    #The categorical kernel needs global optimization urgently
    if estimator.join_kernel == "Cat":
        print("""
        nlopt_opt gopt = nlopt_create(NLOPT_GN_MLSL,%s);
        nlopt_set_lower_bounds(gopt,lb);
        nlopt_set_upper_bounds(gopt,ub);
        nlopt_set_min_objective(gopt,obj,&p);
    """ % (len(Utils.flatten(estimator.kernels))), file=cf)
        print(file=cf)

        print("""
        nlopt_set_maxeval(gopt, %s);
        nlopt_set_ftol_rel(gopt, %s);
        nlopt_opt lopt = nlopt_create(NLOPT_LN_COBYLA,%s);
        nlopt_set_lower_bounds(lopt,lb);
        nlopt_set_upper_bounds(lopt,ub);
        nlopt_set_local_optimizer(gopt, lopt);
        int grc = nlopt_optimize(gopt, bw, &minf);
        assert(grc >=0);
    """ % (40,"1e-10",len(Utils.flatten(estimator.kernels))), file=cf)
    print("   opt_start = std::chrono::high_resolution_clock::now();", file=cf)
    print("""
            nlopt_opt opt = nlopt_create(NLOPT_LN_COBYLA,%s);
            nlopt_set_lower_bounds(opt,lb);
            nlopt_set_upper_bounds(opt,ub);
            nlopt_set_maxeval(opt, %s);
            nlopt_set_ftol_rel(opt, %s);
            nlopt_set_min_objective(opt,obj,&p);
            p.opt = &opt;
            int frc = nlopt_optimize(opt, bw, &minf);
            assert(frc >=0);
        """ % (len(Utils.flatten(estimator.kernels)), 1000, "1e-5"), file=cf)
    
    #print >> cf, "    ddump(bw, %s, bwstr.c_str());" % len(Utils.flatten(estimator.kernels))

    i=0
    for x,kernels in enumerate(estimator.kernels):
        for y,kernel in enumerate(kernels):
            print("    p.bw_t%s_c%s = bw[%s];" % (x,y,i), file=cf)
            i += 1
    print(file=cf)

    
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
        
        print("//", file=cf)
        graph.generateTableEstimateKernel(cf,query,estimator,stats)
        generateBinarySearchCode(cf)
        generateJoinEstimateKernel(cf,query,estimator,stats)
        print("//", file=cf)
                
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
        generateGPUJKDEEstimateFunction(cf,graph,query,estimator,prod(list(ts.values()))**-1.0,stats,cu_factor)
        generateGPUJKDETestWrapper(cf,query,estimator)
        
        if estimator.bw_optimization == "local":
            for tid,table in enumerate(query.tables):
                generateTableEstGrad(cf,tid,query,estimator)
                generateTableObjectiveGrad(cf,tid,query,estimator)
        elif estimator.bw_optimization == "join":
                generateGPUJKDEObjective(cf,query,estimator)
        
        cols = Utils.generateInvariantColumns(query)
        jcols = Utils.generateJoinColumns(query)
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
        print("    p.iteration = (unsigned int) atoi(argv[%s]);" % (len(query.tables)+1), file=cf)
        print("    iteration_stream << \"./iteration\" << std::setw(2) << std::setfill('0') << argv[%s];" % (len(query.tables)+1), file=cf)
        for j,t in enumerate(query.tables):
            print("    p.ss%s = atoi(argv[%s]);" % (j,j+1), file=cf)
            print("    p.ts%s= %s;" % (j,ts[j]), file=cf)
            for k,c in enumerate(t.columns):
                print("    std::stringstream s_t%s_c%s_stream ;" % (j,k), file=cf)
                print("    s_t%s_c%s_stream << iteration_stream.str() << \"/sample_\" << atoi(argv[%s]) << \"_%s_%s.dump\";" % (j,k,j+1,t.tid,c.cid), file=cf) 
                print("    std::string s_t%s_c%s_string = s_t%s_c%s_stream.str();" % (j,k,j,k), file=cf)
                print("    unsigned int* s_t%s_c%s = readUArrayFromFile(s_t%s_c%s_string.c_str());" % (j,k,j,k), file=cf)
                print("    p.s_t%s_c%s = toGPUVector(s_t%s_c%s, p.ss%s, p.ctx, p.queue);" % (j,k,j,k,j), file=cf)
                if estimator.kernels[j][k] == "GaussPoint" or estimator.kernels[j][k] == "GaussRange":
                    print("    p.bw_t%s_c%s = scott_bw(s_t%s_c%s,  p.ss%s, %s);" % (j,k,j,k,j,len(query.tables)), file=cf)
                    print("    if(p.bw_t%s_c%s < 0.2) p.bw_t%s_c%s = 0.2;" % (j,k,j,k), file=cf)
                else:
                    print("    p.bw_t%s_c%s = 1.0/(1.0+1.0/%f);" % (j,k,dv[j][k]-1), file=cf)
            print(file=cf)

        for t,cs in enumerate(jcols):
            if cols[t]:
                for c in cs:
                    print("    p.sr_t%s_c%s = compute::vector<unsigned int>(p.ss%s, p.ctx);" % (t,c,t), file=cf)
        print("    p.final_contributions = compute::vector<double>(p.ss%s, p.ctx);" % (tids[0]), file=cf)
        print("""
    compute::program pr = compute::program::create_with_source(str,p.ctx);
    try{
        std::ostringstream oss;
        pr.build(oss.str());
    } catch(const std::exception& ex){
        std::cout << pr.build_log() << std::endl;
    }
        """, file=cf)
        for j,t in enumerate(query.tables):
            if len(cols[j]) > 0:
                print("    p.invk%s = pr.create_kernel(\"invk_t%s\");" % (j,j), file=cf)   
                print("    p.inv_t%s = compute::vector<double>(p.ss%s, p.ctx);" % (j,j), file=cf)
                print("   p.invr_t%s = compute::vector<double>(p.ss%s, p.ctx);" % (j, j), file=cf)
        print("    p.estimate = pr.create_kernel(\"estimate\");", file=cf)
        print(file=cf)

        for t, tab in enumerate(query.tables):
            print("    p.map_t%s = compute::vector<unsigned int >(p.ss%s+1, p.ctx);" % (t,t), file=cf)
            print("    p.count_t%s = compute::vector<int >(p.ss%s+1, p.ctx);" % (t,t), file=cf)
            print("    p.count_t%s[0] = -1;" % t, file=cf)


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
            
        print("    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";", file=cf)
        print("    p.j_test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());", file=cf)
        
        for i,indices in enumerate(cols):
            if len(indices) != 0:
                for j in indices:
                    if estimator.kernels[i][j] == "GaussRange":
                        print("    std::string j_l_t%s_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                        print("    p.j_l_t%s_c%s= readUArrayFromFile(j_l_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                        print("    std::string j_u_t%s_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                        print("    p.j_u_t%s_c%s = readUArrayFromFile(j_u_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                    else:
                        print("    std::string j_p_t%s_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                        print("    p.j_p_t%s_c%s = readUArrayFromFile(j_p_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
        
        print(file=cf)
        print("    join_test(&p);", file=cf)
        print("}", file=cf)   
        

def generateGPUJKDEObjective(f,query,estimator):
    icols = Utils.generateInvariantColumns(query)
    print("double obj(unsigned n, const double* bw, double* grad, void* f_data){", file=f)

    print("    parameters* p = (parameters*) f_data;", file=f)

    i = 0
    for x,table in enumerate(query.tables):
        for y,col in enumerate(table.columns):
            print("    p->bw_t%s_c%s = bw[%s];" % (x,y,i), file=f)
            i += 1

    print("    int first = 1;", file=f)
    print("    double est = 0.0;", file=f)
    if estimator.objective == "squared":
        print("    double objective = 0.0;", file=f)
    elif estimator.objective == "Q":
        print("    double objective = 1.0;", file=f)
    else:
        raise Exception("I don't know this objective function.")

    print("    for(unsigned int i = 0; i < %s; i++){" % estimator.training, file=f)
    if hasattr(estimator, 'limit_opt'):
        print("    if(std::chrono::duration_cast<std::chrono::minutes>(std::chrono::high_resolution_clock::now()-opt_start).count() > %s)" % estimator.limit_opt, file=f)
        print("        nlopt_force_stop(*(p->opt));", file=f)
    print("            est =  join_estimate_instance(p ", end=' ', file=f)
    for x, cols in enumerate(icols):
        for y in cols: 
            if estimator.kernels[x][y] == "GaussRange":
                print(", p->j_training_u_t%s_c%s[i], p->j_training_l_t%s_c%s[i] " % (x,y,x,y), end=' ', file=f)
            else:
                print(", p->j_training_p_t%s_c%s[i] " % (x,y), end=' ', file=f)
    print(");", file=f)
    print("        unsigned int trues =  p->j_training_cardinality[i];", file=f)

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
        
        
#Generate parameter struct that is passed to the estimation/gradient functions
def generateGPUJKDEParameterArray(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    jcols = Utils.generateJoinColumns(query)
    print("""
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
""", file=f)
    print("    unsigned int iteration;", file=f)
    print("    nlopt_opt* opt;", file=f)
    for j,t in enumerate(query.tables):
        print("    size_t ss%s;" % (j), file=f) 
        print("    unsigned int ts%s;" % (j), file=f)
        if len(cols[j]) > 0:
            print("    compute::kernel invk%s;" % (j), file=f)
            print("    compute::vector<double> inv_t%s;" % (j), file=f)
            print("    compute::vector<double> invr_t%s;" % (j), file=f)
        for k,c in enumerate(t.columns):
            print("    compute::vector<unsigned int> s_t%s_c%s;" % (j,k), file=f)
            print("    double bw_t%s_c%s;" % (j,k), file=f)
        print(file=f)
    print("    compute::kernel estimate;", file=f)

    for t,tab in enumerate(query.tables):
        print("    compute::vector<unsigned int> map_t%s;" % t, file=f)
        print("    compute::vector<int> count_t%s;" % t, file=f)

    for t,_ in enumerate(jcols):
        for c in jcols[t]:
            print("    compute::vector<unsigned int> sr_t%s_c%s;" % (t, c), file=f)
    print("    compute::vector<double> final_contributions;", file=f)
    #Training
    if estimator.bw_optimization == "local":
        for tid,tab in enumerate(query.tables):
            print("    size_t global_t%s;" % (tid), file=f)
            print("    size_t local_t%s;" % (tid), file=f)
            print("    compute::kernel estk%s;" % (tid), file=f)
            print("    compute::kernel gradk%s;" % (tid), file=f)
            print("    compute::vector<double> est_t%s;" % (tid), file=f)
            print("    unsigned int* true_t%s;" % tid, file=f) 
            for cid,col in enumerate(tab.columns):
                print("    compute::vector<double> grad_t%s_c%s;" % (tid,cid), file=f)
                if estimator.kernels[tid][cid] == "GaussRange":
                    print("    unsigned int* l_t%s_c%s;" % (tid,cid), file=f)
                    print("    unsigned int* u_t%s_c%s;" % (tid,cid), file=f)
                else:
                    print("    unsigned int* p_t%s_c%s;" % (tid,cid), file=f)

    print(file=f)    
    print("    unsigned int* j_test_cardinality;", file=f)
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        if len(indices) != 0:
            for j in indices:
                if estimator.kernels[i][j] == "GaussRange":
                    print("    unsigned int* j_l_t%s_c%s;" % (i,j), file=f)
                    print("    unsigned int* j_u_t%s_c%s;" % (i,j), file=f)
                else:
                    print("    unsigned int* j_p_t%s_c%s;" % (i,j), file=f)
    if estimator.bw_optimization == "join":
        print("    unsigned int* j_training_cardinality;", file=f)
                
        for i,indices in enumerate(cols):
        #Start with computing the invariant contributions   
            if len(indices) != 0:
                for j in indices:
                    if estimator.kernels[i][j] == "GaussRange":
                        print("    unsigned int* j_training_l_t%s_c%s;" % (i,j), file=f)
                        print("    unsigned int* j_training_u_t%s_c%s;" % (i,j), file=f)
                    else:
                        print("    unsigned int* j_training_p_t%s_c%s;" % (i,j), file=f)
                
    print("""
} parameters;
""", file=f)


def generateGPUJKDEEstimateFunction(f, nodes, query, estimator, limit, stats, cu_factor):
     icols = Utils.generateInvariantColumns(query)
     jcols = Utils.generateJoinColumns(query)
     ts, dv = stats

     print("double join_estimate_instance(parameters* p", file=f)
     for i, indices in enumerate(icols):
         # Start with computing the invariant contributions
         if len(indices) != 0:
             for j in indices:
                 if estimator.kernels[i][j] == "GaussRange":
                     print("    , unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (i, j, i, j), file=f)
                 else:
                     print("    , unsigned int p_t%s_c%s" % (i, j), file=f)
     print(file=f)
     print("){", file=f)

     nodes.generateTableCode(f, query, estimator, limit, cu_factor)

     if estimator.join_kernel == "Cat":
         if len(query.joins) > 1 or len(query.joins[0]) != 2:
             raise Exception("The categorical kernel does not support more than two joins.")

         # Compute omega_2
         t2, _ = nodes.right_col
         print("    double omega_2 = 1.0;", file=f)
         if icols[t2]:
             print("    boost::compute::reduce(p->invr_t%s.begin(),p->invr_t%s.begin()+rss_t%s, &omega_2, p->queue);" % (
             t2, t2, t2), file=f)
             print("    p->queue.finish();", file=f)

     # Next, generate the limits
     pairs = nodes.collectJoinPairs()
     tids = nodes.collectTableIDs()

     for t1, c1, t2, c2 in pairs:
         if estimator.join_kernel == "Cont":
             print("    double limit_t%s_c%s_t%s_c%s =  compute_cont_limit(p->bw_t%s_c%s, p->bw_t%s_c%s, p->ts%s, p->ts%s);" % (
             t1, c1, t2, c2, t1, c1, t2, c2, t1, t2), file=f)
         elif estimator.join_kernel == "Rect":
             print("    double limit_t%s_c%s_t%s_c%s =  compute_rect_limit(p->bw_t%s_c%s, p->bw_t%s_c%s);" % (
             t1, c1, t2, c2, t1, c1, t2, c2), file=f)
         elif estimator.join_kernel == "Cat":
             print("    double limit_t%s_c%s_t%s_c%s =  0.0;" % (t1, c1, t2, c2), file=f)
         else:
             raise Exception("Unsupported join kernel.")

     print("    size_t local = 64;", file=f)
     print("    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*%s , ((rss_t%s-1)/local+1)*local);" % (cu_factor,tids[0]), file=f)
     print("    local = std::min(local,global);", file=f)

     print("    p->estimate.set_args(", end=' ', file=f)

     for x, t in enumerate(tids):
         for jc in jcols[t]:
             if icols[t]:
                 print("    p->sr_t%s_c%s," % (t, jc), end=' ', file=f)
             else:
                 print("    p->s_t%s_c%s," % (t, jc), end=' ', file=f)
             print("    p->bw_t%s_c%s," % (t, jc), file=f)
         if icols[t]:
             print("    p->invr_t%s," % (t), file=f)
         if x > 0:
             print("    (unsigned int) rss_t%s," % (t), file=f)
     # Here we go.
     for t1, c1, t2, c2 in pairs:
         print("    limit_t%s_c%s_t%s_c%s," % (t1, c1, t2, c2), file=f)
     if estimator.join_kernel == "Cat":
         print("    omega_2,", file=f)
     print("    p->final_contributions, (unsigned int) rss_t%s);" % tids[0], file=f)
     print("    p->queue.enqueue_nd_range_kernel(p->estimate,1,NULL,&(global), &(local));", file=f)

     print("    double est = 0.0;", file=f)
     print("    boost::compute::reduce(p->final_contributions.begin(), p->final_contributions.begin()+std::min(global,rss_t%s), &est, p->queue);" % (
     tids[0]), file=f)
     print("    p->queue.finish();", file=f)
     for i, _ in enumerate(query.tables):
         print("    est *= ((double) p->ts%s)/p->ss%s;" % (i, i), file=f)
     print("    return est;", file=f)
     print("}", file=f)
