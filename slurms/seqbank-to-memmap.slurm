#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=memmap
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 100:00:00
 
#SBATCH --mem-per-cpu=1T
 
# SBATCH -p cascade
#SBATCH -p bigmem
# SBATCH -p gpu-a100-short,gpu-a100,gpu-a100-preempt
# SBATCH --gres=gpu:2
#SBATCH --account=punim2199
 
# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
 
export HF_HOME=/home/rturnbull/wytamma/huggingface

source /home/rturnbull/wytamma/rob/barbet-lightning/.venv/bin/activate

python seqbank-to-memmap.py /data/projects/punim2199/rob/release220/bac120/esm6/esm6.sb /data/projects/punim2199/rob/release220/bac120/esm6/esm6.np /data/projects/punim2199/rob/release220/bac120/esm6/esm6.txt
