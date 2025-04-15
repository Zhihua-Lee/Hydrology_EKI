#!/bin/sh
#$ -N Itest10_201606
#$ -j y
#$ -cwd
#$ -pe smp 28
#$ -l mf=16G
#$ -q IFC
#$ -m es
#$ -M zli333@uiowa.edu

/bin/echo Running on host: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

# modules must be consistent with compiling
module reset
module load openmpi

mkdir /nfsscratch/Users/zli333/test
mpirun -np 28 /Users/zli333/DA/2025_EKI/exec/asynch/bin/asynch /Users/zli333/DA/2025_EKI/Inverse_Problem/hlm_data/Simulated_data/presimulate_hydrograph/template_pre_sim.gbl

#moved to post_processing python3 /Users/zli333/network_conditioning/codes/dats2parquet.py /Users/zli333/DA/Zhihua/hlm_runs/test10/dats/ /Users/zli333/DA/Zhihua/hlm_runs/test10/sim_flows.gzip -r
rm -r /nfsscratch/Users/zli333/test