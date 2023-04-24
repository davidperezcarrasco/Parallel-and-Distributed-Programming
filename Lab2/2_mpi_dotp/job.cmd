#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_dotp.out
#SBATCH --error=out_dotp.err
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00

source /etc/profile.d/z00-global-profile.sh

module load GCC/10.2.0
module load OpenMPI/4.1.0-GCC-10.2.0


make clean >> make.out && make >> make.out || exit 1 


mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=4 ./a.out
