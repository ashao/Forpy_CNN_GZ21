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
#SBATCH --nodes=5               # node count
#SBATCH --ntasks-per-node=128    # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G

source /home/aeshao/mom6_smartsim_env_gnu.sh
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aeshao/dev/gnu/SmartRedis/install/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aeshao/dev/SmartRedis/install/lib
ulimit -s unlimited

#python call_double_gyre_clustered.py

python call_OM4_025_clustered.py
