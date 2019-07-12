#!/bin/bash
#SBATCH --job-name=cifar10-ddp
#SBATCH --output=logs.out
#SBATCH --error=logs.err
#SBATCH --gres gpu:2
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2


export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

set -x
srun python -u -m ddp_example.train_cifar10
