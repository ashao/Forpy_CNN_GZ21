#!/bin/bash
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu	         #
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=1     # number of tasks per node
#SBATCH --cpus-per-task=64        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # use GPU
#SBATCH --mem=32G
#SBATCH hetjob
#SBATCH --partition=cimes
#SBATCH --nodes=16               # node count
#SBATCH --ntasks-per-node=128    # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G

# Equivalent to
# salloc --time=02:00:00 --partition=gpu --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:1 --mem=32G : --partition=cimes --nodes=1 --ntasks-per-node=128 --cpus-per-task=1 --mem=256G

source /home/aeshao/mom6_smartsim_env.sh
ulimit -s unlimited

#python call_double_gyre_clustered.py

python call_OM4_025_clustered.py
