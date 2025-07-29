#!/bin/sh
#$ -N Presimulate_using_Cr_ref
#$ -j y
#$ -cwd
#$ -pe smp 112
#$ -l mf=16G
#$ -q IFC
#$ -m es
#$ -M zli333@uiowa.edu
#$ -o /dev/null
#$ -e /dev/null

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

module reset
module load openmpi

mkdir -p /nfsscratch/Users/zli333/test
mpirun -np 112 /Users/zli333/DA/2025_EKI/exec/asynch/bin/asynch hlm_data/Simulated_data/presimulate_hydrograph/presim_run/presim.gbl
rm -r /nfsscratch/Users/zli333/test
