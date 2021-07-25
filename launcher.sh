#!/bin/bash
#SBATCH --job-name=cifar10-ddp
#SBATCH --output=logs.out
#SBATCH --error=logs.err
#SBATCH --gpus-per-task 1  # This gives 1 GPU to each process (or task)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2 # The number of processes for slurm to start on each node
#SBATCH --partition=short


export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
srun -u python -u -m ddp_example.train_cifar10
