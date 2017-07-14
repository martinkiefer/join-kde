# join-kde
Proof of concept for join cardinality estimation using GPU-accelerated Kernel Density Estimator models. This repository contains all our experiment code for our estimators, baseline estimators, and datasets. We use code generation to create the estimators and their respective GPU.

## Requirements
1. The GNU Compiler Collection (GCC) (gcc, g++).
2. A recent version of Postgres. We used Postgres 9.6.3.
3. A recent version of the nlopt library. We used version 2.4.2.
4. Python 2.7 and the following modules:
⋅⋅*  scipy
⋅⋅*  numpy
⋅⋅*  psycopg2
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
TODO
