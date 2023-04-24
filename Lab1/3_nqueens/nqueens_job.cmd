#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=out_nqueens.out
#SBATCH --error=out_nqueens.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:00:10

source /etc/profile.d/z00-global-profile.sh

module load GCC/10.2.0

make clean >> out_nqueens.err && make >> out_nqueens.err || exit 1 

./a.out 100 100 100

