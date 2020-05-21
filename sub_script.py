#!/bin/bash
#PBS -j oe
#PBS -l ncpus=16
#PBS -l mpiprocs=16
#PBS -l mem=96gb

module load singularity
cd /home/elefongx/proj/fpe/fipy/Ray_testing
singularity exec /opt/containers/ubuntu/18/fipy_v2_2a.sif python ./Ray_Phi_Integration_modified.py
