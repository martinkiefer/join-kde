# join-kde
Proof of concept for join cardinality estimation using GPU-accelerated Kernel Density Estimator models. This repository contains all our experiment code for our estimators, baseline estimators, and datasets. We use code generation to create the estimator host code (C++) and device code (OpenCL).

## Requirements
1. The GNU Compiler Collection (GCC) (gcc, g++).
2. A recent version of Postgres. We used Postgres 9.6.3.
3. A recent version of the nlopt library. We used version 2.4.2.
4. Python3 and the following modules:
  *  scipy
  *  numpy
  *  psycopg2
5. A recent version of the boost C++ library. We used version 1.64.0.
6. A GPU that supports OpenCL and an according OpenCL SDK. An OpenCL-capable CPU and an according OpenCL SDK will do as well, but note that our code is designed for GPUs.
7. A recent version of gnuplot is required to create experiment plots. We used version 5.0 (Patchlevel 6).
8. A Linux-based operating system. While the code may compile using OSX as well, the Apple OpenCL SDK is buggy and will lead to incorrect results.

## Setup
1. Create an initial database folder and an initial database using the initdb and the createdb command.
2. Download the dump containing our datasets from here: https://blog.boxm.de/dump2.gz
3. Use the psql command to load our experimental data into the database.

zcat dump2.gz | psql

## Running Experiments
Our experiments are centered around the concept of experiment descriptors. These query descriptors are JSON files that define the entire experiment. The query descriptors for our experiments are located in the folder experiment_descriptor. 

To run an experiment perform the following steps:
1. Copy the experiment descriptor of your choice to code/descriptor.json
2. Run the run_experiment.sh script.

The code will create the experiment workload, samples, estimator code and auxilliary scripts. After that, the estimators are compiled and the experiment is started. 

Note that you have to change the variables pg_conf and compiler_options in the experiment descriptor to match your system setup. A description of the most important configuration variables is given in the next secion.

The code will execute the experiment on the default OpenCL device. If you want to use a different device, you can use the environment variables BOOST_COMPUTE_DEFAULT_DEVICE (device name) or BOOST_COMPUTE_DEFAULT_DEVICE_TYPE (GPU, CPU).

Note that a large experiment can take a while, so be sure to run the experiment using nohup or in a tmux session.

After the experiment is finished, the estimation results for every test query and every iteration are located in CSV files with the suffix .results. They have the following format:

iteration, model_size, estimated cardinality, true cardinality, squared error, absolute error, relative error, Q error, execution time in nanoseconds


## Experiment Descriptors
The experiment descriptors define an entire experiment, the most important variables to use or adjust our experiments are:

### System Parameters
*pg_conf*: String, Contains the connection information for your Postgres instance. Passed to Psycopg2.

*compiler_options, String:* Commandline parameters for g++. Make sure the compiler has all necessary flags to use nlopt, OpenCL and Boost.

*compute_unit_factor, int* Controls the oversubscription per compute unit for OpenCL. The current value is fine for GPUs, but can be set to 64 for CPUs. Note that our code is not optimized for CPUs, though.

### Global Experiment Parameters
*iterations, int:* Number of times the experiment is repeated with different samples, queries, seeds etc.

*size_type, String:* "relative" or "absolute". Controls whether the model size parameters are given relative to base table sizes or absolute (Bytes).

*model_size, List:* List of numeric types. Specify all model sizes you want to evaluate per iteration.

*workload, String:* "mixed_distinct" or "mixed_uniform" switches between the distinct and the uniform workload.

*genertate_table_samples, boolean:* Do we need uniform table samples for the experiment? 

*generate_correlated_samples, boolean:* Do we need correlated table samples for the experiment? 

*generate_join_samples, boolean:* Do we need uniform join samples for this experiment?

*dump_tables, boolean:* Do we need table dumps for this experiment (AGMS)?

*generate_join_training_queries, int:* Number of training queries.

*generate_join_test_queries, int:* Number of test queries.

*generate_table_X_queries, int:* Set to zero. Deprecated.

### The query_descriptor Object
The query descriptor contains all the information to describe the involved tables, joins and base table selections. Looking at the examples in the experiment_descriptors folder this should be self-explanatory to the largest extent.

*tables, List of table descriptors:* Table descriptors for all involved tables.

*table_descriptor:* Contains all the information for a table. Includes the name (tid) as well as a list of column descriptor (columns).

*column_descriptor:* Contains all the information on a column. Includes the column name (cid) and the type of predicate for the column (type: point, range). Choose "point" for join attributes.

*joins, list of lists of [table_offset, column_offset] pairs:* Describes the joins between the tables. Every join equivalence class is a sublist. Each of these sublists contains two or more [table_offset, column_offset]. table_offset and column_offset are given as the offset in the tables and columns lists.


### The Estimators List
*estimators, list of estimator objects*: All the estimators that are evaluated in this experiment.


Common attributes are:

*estimator, String:* Name of the estimator. Possible values: AGPUJKDE (also known as TS+KDE), GPUKDE (JS+KDE), GPUSample (TS), GPUJoinSample (JS), AGMS, GPUCorrelatedSample (Correlated Sampling), Postgres

*test, Integer:* Number of training queries for this estimator. Should have the same value for all esitmators. Should be less or equal than generate_join_test_queries.


Attributes for GPUKDE, AGPUJKDE:

*objective, string:* "Q" for the multiplicative error. "squared" is deprecated.

*bw_optimization, String:* "join" is the only supported training method right now.

*join_kernel, String:* "Cont" is the only supported join kernel right now (AGPUKDE only)

*limit_opt, int:* Timeout in minutes for the bandwidth optimization performed by nlopt (AGPUJKDE only)

*test, Integer:* Number of test queries for this estimator.

*look_behind, boolean:* False. Deprecated.

*kernels, list of lists of strings:* Specifies the kernels for table and attributes in order of appearance in the query descriptor. Every table is a sublist. Possible values are "GaussPoint" for attributes with "point" attributes in the query descriptor and "GaussRange" for "range" attributes in the query descriptor.

## Experiment Visualization
We provide scripts to visualize the experiments as well. The folder visualization contains a set of scripts for estiation quality experiments with statitc model size (static-quality), estimation quality experiments with varying model size (scaling-quality), and runtime experiments with varying model size (scaling-runtime).

The code folder needs to be copied or sym-linked into the corresponing directory. For example, to visualize a static quality experiment perform the following steps:

1. Copy or symlink the code folder to static-quality/static/code
2. Execute the plot.sh script while being in the static-quality directory.
3. The plot can be found in static-quality/static.pdf

You can also create multiple plots at once by creating copies of the static folder for each experiment and copying the respective code folder to these directories. The plot will have the same name as the containing directory.
