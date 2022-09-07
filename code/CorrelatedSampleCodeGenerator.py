 #Code generator for gauss kernel
import Utils
from KDECodeGenerator import generatePreamble
from SampleCodeGenerator import generateBinarySearchCode
from SampleCodeGenerator import generateCIncludes
from SampleCodeGenerator import generateJoinEstimateKernel
from SampleCodeGenerator import prod
from JoinGraph import constructJoinGraph
        

def generateGPUSampleCode(i,query,estimator,stats,cu_factor):
    ts, dv = stats
    graph = constructJoinGraph(query)
    tids = graph.collectTableIDs()

    #Generate Kernels    
    with open("./%s_kernels.cl" % i,'w') as cf:
        generatePreamble(cf)
        
        print("//", file=cf)
        graph.generateTableEstimateKernel(cf, query, estimator, stats)
        generateBinarySearchCode(cf)
        generateJoinEstimateKernel(cf,query,estimator,stats)
        print("//", file=cf)

        
    with open("./%s_GPUCS.cpp" % i,'w') as cf:
        generateCIncludes(cf)
        print("size_t res_size = 0;", file=cf)
        generateGPUCorrelatedSampleParameterArray(cf,query,estimator)
        Utils.generateGPUVectorConverterFunction(cf)
        Utils.generateUintFileReaderFunction(cf)
        Utils.generateScottBWFunction(cf)
        generateGPUSampleEstimateFunction(cf,graph,query,estimator,prod(list(ts.values()))**-1.0,stats,cu_factor)
        #There is no reason why we shouldn't use the estimate function from GPUJKDE.
        generateCSTestWrapper(cf,query,estimator)
        
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
            print("    p.ts%s= %s;" % (j,ts[j]), file=cf)
            print("    p.target_ss%s= atoi(argv[%s]);" % (j,j+1), file=cf)
            for k,c in enumerate(t.columns):
                print("    std::stringstream s_t%s_c%s_stream ;" % (j,k), file=cf)
                print("    s_t%s_c%s_stream << iteration_stream.str() << \"/csample_\" << atoi(argv[%s]) << \"_%s_%s.dump\";" % (j,k,j+1,t.tid,c.cid), file=cf) 
                print("    std::string s_t%s_c%s_string = s_t%s_c%s_stream.str();" % (j,k,j,k), file=cf)
                print("    unsigned int* s_t%s_c%s = readUArrayFromFile(s_t%s_c%s_string.c_str(),&p.ss%s);" % (j,k,j,k,j), file=cf)
                print("    p.ss%s /= sizeof(unsigned int);" % (j), file=cf)
                print("    p.s_t%s_c%s = toGPUVector(s_t%s_c%s, p.ss%s, p.ctx, p.queue);" % (j,k,j,k,j), file=cf)
            print("    res_size += atoi(argv[%s]);" % (j+1), file=cf)
            print(file=cf)

        for t,cs in enumerate(jcols):
            if cols[t]:
                for c in cs:
                    print("    p.sr_t%s_c%s = compute::vector<unsigned int>(p.ss%s, p.ctx);" % (t,c,t), file=cf)
        print("    p.final_contributions = compute::vector<unsigned long>(p.ss%s, p.ctx);" % tids[0], file=cf)
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
            
        print("    std::string test_cardinality_string = iteration_stream.str() + \"/test_join_true.dump\";", file=cf)
        print("    p.j_test_cardinality = readUArrayFromFile(test_cardinality_string.c_str());", file=cf)
        
        for i,indices in enumerate(cols):
            if len(indices) != 0:
                for j in indices:
                    if query.tables[i].columns[j].type == "range":
                        print("    std::string j_l_t%s_c%s_string = iteration_stream.str() + \"/test_join_l_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                        print("    p.j_l_t%s_c%s= readUArrayFromFile(j_l_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                        print("    std::string j_u_t%s_c%s_string = iteration_stream.str() + \"/test_join_u_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                        print("    p.j_u_t%s_c%s = readUArrayFromFile(j_u_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                    elif query.tables[i].columns[j].type == "point":
                        print("    std::string j_p_t%s_c%s_string = iteration_stream.str() + \"/test_join_p_%s_%s.dump\";" % (i,j,query.tables[i].tid,query.tables[i].columns[j].cid), file=cf)
                        print("    p.j_p_t%s_c%s = readUArrayFromFile(j_p_t%s_c%s_string.c_str());" % (i,j,i,j), file=cf)
                    else:
                        raise Exception("Unsupported ctype.")
        
        print(file=cf)
        print("    join_test(&p);", file=cf)
        print("}", file=cf)   

def generateGPUSampleEstimateFunction(f, nodes, query, estimator, limit, stats, cu_factor):
     icols = Utils.generateInvariantColumns(query)
     jcols = Utils.generateJoinColumns(query)
     ts, dv = stats

     print("double join_estimate_instance(parameters* p", file=f)
     for i, indices in enumerate(icols):
         # Start with computing the invariant contributions
         if len(indices) != 0:
             for j in indices:
                 if query.tables[i].columns[j].type == "range":
                     print("    , unsigned int u_t%s_c%s, unsigned int l_t%s_c%s" % (i, j, i, j), file=f)
                 elif query.tables[i].columns[j].type == "point":
                     print("    , unsigned int p_t%s_c%s" % (i, j), file=f)
                 else:
                     raise Exception("Unknown ctype.")
     print(file=f)
     print("){", file=f)

     nodes.generateTableCode(f, query, estimator, limit, cu_factor)

     # Next, generate the limits
     pairs = nodes.collectJoinPairs()
     tids = nodes.collectTableIDs()


     print("    size_t local = 64;", file=f)
     print("    size_t global = std::min((size_t) p->ctx.get_device().compute_units()*%s, ((rss_t%s-1)/local+1)*local);" % (cu_factor,tids[0]), file=f)
     print("    local = std::min(local,global);", file=f)
     print("    p->estimate.set_args(", end=' ', file=f)

     for x, t in enumerate(tids):
         for jc in jcols[t]:
             if icols[t]:
                 print("    p->sr_t%s_c%s," % (t, jc), end=' ', file=f)
             else:
                 print("    p->s_t%s_c%s," % (t, jc), end=' ', file=f)
         if x > 0:
             print("    (unsigned int) rss_t%s," % (t), file=f)
     print("    p->final_contributions, (unsigned int) rss_t%s);" % tids[0], file=f)
     print("    p->queue.enqueue_nd_range_kernel(p->estimate,1,NULL,&(global), &(local));", file=f) 

     print("    unsigned long counter = 0.0;", file=f)
     print("    boost::compute::reduce(p->final_contributions.begin(), p->final_contributions.begin()+std::min(rss_t%s,global), &counter, p->queue);" % (
     tids[0]), file=f)
     print("    p->queue.finish();", file=f)
     print("    double est = counter;", file=f)
     for jid,join in enumerate(query.joins):
         print("    double p%s = 1.0;" % (jid), end=' ', file=f)
         for tid, cid in join:
             print("    p%s = std::min(p%s, pow(((double) p->target_ss%s) / p->ts%s, 1.0 / %s));" % (jid,jid,tid,tid,len(join)), end=' ', file=f)
         print(" est /= p%s;" % (jid), file=f)
     print("    return est;", file=f)
     print("}", file=f)


def generateCSTestWrapper(f,query,estimator):
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
    Utils.printObjectiveLine(f,"res_size")
    print("    }", file=f)
    print("    return objective/%s;" % estimator.test, file=f)
    print("}", file=f)

#Generate parameter struct that is passed to the estimation/gradient functions
def generateGPUCorrelatedSampleParameterArray(f,query,estimator):
    cols = Utils.generateInvariantColumns(query)
    jcols = Utils.generateJoinColumns(query)
    print("""
typedef struct{
    compute::command_queue queue;
    compute::context ctx;
""", file=f)
    print("    unsigned int iteration;", file=f)
    for j,t in enumerate(query.tables):
        print("    size_t ss%s;" % (j), file=f) 
        print("    size_t target_ss%s;" % (j), file=f) 
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
    print("    compute::vector<unsigned long> final_contributions;", file=f)
    #Training

    print(file=f)    
    print("    unsigned int* j_test_cardinality;", file=f)
    for i,indices in enumerate(cols):
    #Start with computing the invariant contributions   
        if len(indices) != 0:
            for j in indices:
                if query.tables[i].columns[j].type == "range":
                    print("    unsigned int* j_l_t%s_c%s;" % (i,j), file=f)
                    print("    unsigned int* j_u_t%s_c%s;" % (i,j), file=f)
                elif query.tables[i].columns[j].type == "point":
                    print("    unsigned int* j_p_t%s_c%s;" % (i,j), file=f)
                else:
                    raise Exception("Unknown ctype.")
                
    print("""
} parameters;
""", file=f)


