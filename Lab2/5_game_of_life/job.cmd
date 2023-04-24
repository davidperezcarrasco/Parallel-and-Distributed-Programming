#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_game.out
#SBATCH --error=out_game.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:10

source /etc/profile.d/z00-global-profile.sh

module load GCC/10.2.0
module load OpenMPI/4.1.0-GCC-10.2.0


make clean >> make.out && make >> make.out || exit 1 

mpirun -np 1 ./a.out
mpirun -np 8 ./a.out
