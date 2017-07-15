# join-kde
Proof of concept for join cardinality estimation using GPU-accelerated Kernel Density Estimator models. This repository contains all our experiment code for our estimators, baseline estimators, and datasets. We use code generation to create the estimator host code (C++) and device code (OpenCL)

## Requirements
1. The GNU Compiler Collection (GCC) (gcc, g++).
2. A recent version of Postgres. We used Postgres 9.6.3.
3. A recent version of the nlopt library. We used version 2.4.2.
4. Python 2.7 and the following modules:
  *  scipy
  *  numpy
  *  psycopg2
5. A recent version of the boost C++ library. We used version 1.64.0.
6. A GPU that supports OpenCL and an according OpenCL SDK. An OpenCL-capable CPU and an according OpenCL SDK will do as well, but note that our code is designed for GPUs.

## Setup
1. Create an initial database folder and an initial database using the initdb and the createdb command.
2. Use the psql command to load our experimental data into the database.

zcat dump.gz | psql

## Running Experiments
Our experiments are centered around the concept of experiment descriptors. These query descriptors are JSON files that define the entire experiment. The query descriptors for our experiments are located in the folder experiment_descriptor. 

To run an experiment perform the following steps:
1. Copy the experiment descriptor of your choice to code/descriptor.json
2. Run the run_experiment.sh script.

The code will create the experiment workload, samples, estimator code and auxilliary scripts. After that, the estimators are compiled and the experiment is started. 

Note that you have to change the variables pg_conf and compiler_options in the experiment descriptor to match your system setup. A description of the most important configuration variables is given in the next secion.

The code will execute the experiment on the default OpenCL device. If you want to use a different device, you can use the environment variables BOOST_COMPUTE_DEFAULT_DEVICE (device name) or BOOST_COMPUTE_DEFAULT_DEVICE_TYPE (GPU, CPU).


## Experiment Descriptors
The experiment descriptors define an entire experiment, the most important variables to use or adjust our experiments are:

### System Parameters
pg_conf: String, Contains the connection information for your Postgres instance. Passed to Psycopg2.
compiler_options: String, Commandline parameters for g++. Make sure the compiler has all necessary flags to use nlopt, OpenCL and Boost.
compute_unit_factor: int, Controls the oversubscription per compute unit for OpenCL. The current value is fine for GPUs, but can be set to 64 for CPUs. Note that our code is not optimized for CPUs, though.

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
*generate_table_*_queries, int:* Set to zero. Deprecated.

### The query_descriptor object
The query descriptor contains all the information to describe the involved tables, joins and base table selections. Looking at the examples in the experiment_descriptors folder this should be self-explanatory to the largest extent.

*tables, List of table descriptors:* Table descriptors for all involved tables.
*table_descriptor:* Contains all the information for a table. Includes the name (tid) as well as a list of column descriptor (columns).
*column_descriptor:* Contains all the information on a column. Includes the column name (cid) and the type of predicate for the column (type: point, range). Choose "point" for join attributes.

*joins, list of lists of [table_offset, column_offset] pairs:* Describes the joins between the tables. Every join equivalence class is a sublist. Each of these sublists contains two or more [table_offset, column_offset]. table_offset and column_offset are given as the offset in the tables and columns lists.
