import KDECodeGenerator
import JoinSampleCodeGenerator
import SampleCodeGenerator
import CorrelatedSampleCodeGenerator
import AggressiveKDECodeGenerator
import numpy as np
import os.path
import UniformJoinQueryGenerator as ujqg
import MixedJoinQueryGenerator as mjqg
import UniformTableQueryGenerator as utqg
import TableSampleGenerator as tsg
import CorrelatedTableSampleGenerator as ctsg
import JoinSampleGenerator as jsg
import TableDump as td
import AGMSCodeGenerator
import Utils
import pickle
import SizeComputations

   
configuration = None
with open('./descriptor.json', 'r') as f:
    descriptor = Utils.json2obj(f.read())
    
stats = None
jstats = None
pt_sizes = []

#If the model size is given relative to the table size, we need to preprocess the model sizes.
if descriptor.sizetype == "relative":
    #First, we replace all model sizes with their corresponing absolute model size
    stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
    allsize = SizeComputations.ComputeAllTableSizes(descriptor.query_descriptor,stats)
    sumsize = np.sum(allsize)
    
    #Now, we have to create the individual model sizes for per-table samples. Meh.     
    pt_sizes = []
    for i, size in enumerate(descriptor.model_sizes):
        pt_sizes.append((allsize * size).astype(int))        
    
    for i, size in enumerate(descriptor.model_sizes):
        descriptor.model_sizes[i] = int(sumsize*size)   
        
elif descriptor.sizetype == "absolute":
        for i, size in enumerate(descriptor.model_sizes):
            pt_sizes.append(np.array([size/len(descriptor.query_descriptor.tables)]*len(descriptor.query_descriptor.tables)))       
else:
    raise Exception("Invalid model size type option.")
    
cf = open("./compile.sh",'w')
cf.write("#/usr/bin/bash\n")
rf = open("./run.sh",'w')
rf.write("#/usr/bin/bash\n")

#Iterate over estimators and generate the code
for i,estimator in enumerate(descriptor.estimators):
    #continue
    if estimator.estimator == "AGPUJKDE":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        AggressiveKDECodeGenerator.generateGPUJKDECode(i,descriptor.query_descriptor,estimator,stats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_AGPUJKDE.cpp" % i,descriptor.compiler_options,"%s_AGPUJKDE" % i))
        for ss in pt_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_AGPUJKDE" % i, ' '.join(SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,ss).astype(str)), it, "%s_AGPUJKDE.results" % i))
    elif estimator.estimator == "GPUSample":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        SampleCodeGenerator.generateGPUSampleCode(i,descriptor.query_descriptor,estimator,stats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_GPUS.cpp" % i,descriptor.compiler_options,"%s_GPUS" % i))
        for ss in pt_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_GPUS" % i, ' '.join(SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,ss).astype(str)), it, "%s_GPUS.results" % i))
    elif estimator.estimator == "GPUCorrelatedSample":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        CorrelatedSampleCodeGenerator.generateGPUSampleCode(i,descriptor.query_descriptor,estimator,stats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_GPUCS.cpp" % i,descriptor.compiler_options,"%s_GPUCS" % i))
        for ss in pt_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_GPUCS" % i,' '.join(SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,ss).astype(str)), it, "%s_GPUCS.results" % i))
    elif estimator.estimator == "AGPUJKDE_COUNT":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        CountAggressiveKDECodeGenerator.generateGPUJKDECode(i,descriptor.query_descriptor,estimator,stats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_AGPUJKDE.cpp" % i,descriptor.compiler_options,"%s_AGPUJKDE" % i))
        for ss in pt_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_AGPUJKDE" % i, ' '.join(SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,ss).astype(str)), it, "%s_AGPUJKDE.results" % i))
    elif estimator.estimator == "CountSample":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        #For now, its sufficient to pickle the stats
        for ss in pt_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("python2 ./CountSampleEvaluator.py %s %s %s >> %s\n" % (' '.join(SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,ss).astype(str)), it, estimator.test, "%s_Sample.results" % i))     
        with open("./stats.pick",'w') as f:
            pickle.dump(stats,f)
    elif estimator.estimator == "GPUJoinSample":
        if jstats == None:
            jstats = Utils.retreiveJoinStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        JoinSampleCodeGenerator.generateGPUJoinSampleCode(i,descriptor.query_descriptor,estimator,jstats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_GPUJS.cpp" % i,descriptor.compiler_options,"%s_GPUJS" % i))
        for ss in descriptor.model_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_GPUJS" % i, SizeComputations.ComputeGPUKDESampleSize(descriptor.query_descriptor,ss), it, "%s_GPUJS.results" % i))
    elif estimator.estimator == "GPUKDE":
        if jstats == None:
            jstats = Utils.retreiveJoinStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        KDECodeGenerator.generateGPUKDECode(i,descriptor.query_descriptor,estimator,jstats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_GPUKDE.cpp" % i,descriptor.compiler_options,"%s_GPUKDE" % i))
        for ss in descriptor.model_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_GPUKDE" % i, SizeComputations.ComputeGPUKDESampleSize(descriptor.query_descriptor,ss), it, "%s_GPUKDE.results" % i))
    elif estimator.estimator == "AGMS":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        AGMSCodeGenerator.generateAGMSCode(i,descriptor.query_descriptor,estimator,stats,descriptor.compute_unit_factor)
        cf.write("g++ -O2 -std=c++11 %s %s -o %s\n" % ("%s_AGMS.cpp" % i,descriptor.compiler_options,"%s_AGMS" % i))
        for ss in descriptor.model_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("./%s %s %s >> %s\n" % ("%s_AGMS" % i, SizeComputations.ComputeAGMSSkn(descriptor.query_descriptor,ss), it, "%s_AGMS.results" % i))     
    elif estimator.estimator == "Postgres":
        for it in range(0,descriptor.iterations):
            rf.write("python2 ./PostgresEvaluator.py %s %s >> %s_Postgres.results\n" % (it,estimator.test,i))
    elif estimator.estimator == "JoinSample":
        if jstats == None:
            jstats = Utils.retreiveJoinStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        for ss in descriptor.model_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("python2 ./JoinSampleEvaluator.py %s %s %s >> %s\n" % (SizeComputations.ComputeGPUKDESampleSize(descriptor.query_descriptor,ss), it, estimator.test, "%s_JoinSample.results" % i))            
        with open("./jstats.pick",'w') as f:
            pickle.dump(jstats,f)
    elif estimator.estimator == "Sample":
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        #For now, its sufficient to pickle the stats
        for ss in pt_sizes:
            for it in range(0,descriptor.iterations):
                rf.write("python2 ./SampleEvaluator.py %s %s %s >> %s\n" % (' '.join(SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,ss).astype(str)), it, estimator.test, "%s_Sample.results" % i))     
        with open("./stats.pick",'w') as f:
            pickle.dump(stats,f)
    else:
        raise Exception("Unknown estimator: %s" % estimator.estimator)
rf.close()
cf.close()


#size = SizeComputations.ComputeGPUJKDESize(descriptor.query_descriptor,2048)
#print SizeComputations.ComputeAGMSSkn(descriptor.query_descriptor,size)

if not os.path.exists("./data_lock"):
    print("Generating test data...")
    if descriptor.dump_tables:
         td.dumpTables(descriptor.pg_conf,descriptor.query_descriptor,"./","")
    
    if descriptor.generate_join_samples:        
        jsg.generateSamples(descriptor.pg_conf,descriptor.query_descriptor,SizeComputations.ComputeGPUKDESampleSize(descriptor.query_descriptor,np.array(descriptor.model_sizes)),"./","",descriptor.iterations)
        
    if descriptor.generate_table_samples:            
        tsg.generateSamples(descriptor.pg_conf,descriptor.query_descriptor,SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,np.array(pt_sizes)),"./","",descriptor.iterations)

    if descriptor.generate_correlated_table_samples:            
        if stats == None:
            stats = Utils.retreiveTableStatistics(descriptor.pg_conf,descriptor.query_descriptor)
        ctsg.generateCorrelatedSamples(descriptor.pg_conf,descriptor.query_descriptor,SizeComputations.ComputeGPUJKDESampleSize(descriptor.query_descriptor,np.array(pt_sizes)),"./","",descriptor.iterations, stats)

f = open('./data_lock','w')
f.close()

if not os.path.exists("./query_lock"):
    if descriptor.workload == "uniform":
        if descriptor.generate_join_training_queries > 0:
            ujqg.generateQueryData(descriptor.pg_conf,descriptor.query_descriptor,descriptor.generate_join_training_queries,"./","training_",descriptor.iterations)

        if descriptor.generate_join_test_queries > 0:
            ujqg.generateQueryData(descriptor.pg_conf,descriptor.query_descriptor,descriptor.generate_join_test_queries,"./","test_",descriptor.iterations)

        if descriptor.generate_table_training_queries > 0:
            utqg.generateQueryData(descriptor.pg_conf,descriptor.query_descriptor,descriptor.generate_table_training_queries,"./","training_",descriptor.iterations)

        if descriptor.generate_table_test_queries > 0:
            utqg.generateQueryData(descriptor.pg_conf,descriptor.query_descriptor,descriptor.generate_table_test_queries,"./","test_",descriptor.iterations)
    elif descriptor.workload == "distinct":
        if descriptor.generate_join_training_queries > 0:
            djqg.generateQueryData(descriptor.pg_conf, descriptor.query_descriptor,
                                   descriptor.generate_join_training_queries, "./", "training_", descriptor.iterations)

        if descriptor.generate_join_test_queries > 0:
            djqg.generateQueryData(descriptor.pg_conf, descriptor.query_descriptor,
                                   descriptor.generate_join_test_queries, "./", "test_", descriptor.iterations)

        if descriptor.generate_table_training_queries > 0:
            dtqg.generateQueryData(descriptor.pg_conf, descriptor.query_descriptor,
                                   descriptor.generate_table_training_queries, "./", "training_", descriptor.iterations)

        if descriptor.generate_table_test_queries > 0:
            dtqg.generateQueryData(descriptor.pg_conf, descriptor.query_descriptor,
                                   descriptor.generate_table_test_queries, "./", "test_", descriptor.iterations)
    elif descriptor.workload == "mixed_uniform":
        if descriptor.generate_join_training_queries > 0 or descriptor.generate_join_test_queries > 0:
            mjqg.generateQueryData(descriptor.pg_conf,descriptor.query_descriptor,descriptor.generate_join_training_queries,"./","training_",descriptor.generate_join_training_queries,"./","test_",descriptor.iterations,"uniform")
    elif descriptor.workload == "mixed_distinct":
        if descriptor.generate_join_training_queries > 0 or descriptor.generate_join_test_queries > 0:
            mjqg.generateQueryData(descriptor.pg_conf,descriptor.query_descriptor,descriptor.generate_join_training_queries,"./","training_",descriptor.generate_join_training_queries,"./","test_",descriptor.iterations,"distinct")
    else:
        raise Exception("Unknown workload.")
else:
    print("WARNING: There was a generator lock.")

f = open('./query_lock','w')
f.close()
