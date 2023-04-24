#!/bin/bash

#SBATCH --job-name=image_mem
#SBATCH --output=image_%j.out
#SBATCH --error=image_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1             	# Number of gpus
#SBATCH --time=00:10:00

module load NVHPC/21.2

make gpu || exit 1
./imgproc_mem lleo 1920 1214

