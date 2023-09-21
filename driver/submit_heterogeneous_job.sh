#!/bin/bash
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu	         #
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=48     # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # use GPU
#SBATCH hetjob
#SBATCH --partition=cimes
#SBATCH --nodes=5               # node count
#SBATCH --ntasks-per-node=128    # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)

module use --append /scratch/gpfs/aeshao/local/modulefiles
module load intel/2021.1 intel-mpi/intel/2021.1.1
module load cudatoolkit/11.7 cudnn/8.9.1
module load anaconda3/2022.5
module load netcdf/intel-2021.1/hdf5-1.10.6/4.7.4
conda activate smartsim-dev
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aeshao/dev/SmartRedis/install/lib
ulimit -s unlimited

#python call_MOM6_clustered.py
python call_OM4_025_clustered.py
