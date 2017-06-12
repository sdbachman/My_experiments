#!/bin/tcsh
### Job Name
#PBS -N mpi_job
### Project code
#PBS -A NCGD0011
#PBS -l walltime=00:30:00
#PBS -q share
### Merge output and error files
#PBS -j oe
### Select 2 nodes with 36 CPUs each for a total of 72 MPI processes
#PBS -l select=1:ncpus=18:mpiprocs=18
### Send email on abort, begin and end
#PBS -m abe
### Specify mail recipient
#PBS -M bachman@ucar.edu

setenv MPI_USE_ARRAY false

### Run the executable
mpirun -np 16 MOM6
#mpiexec_mpt dplace -s 1 MOM6
