#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=predict
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 4:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-h100,gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --account=punim2199
 
# Specify your email address to be notified of progress.
#SBATCH --mail-user=robert.turnbull@unimelb.edu.au
#SBATCH --mail-type=ALL
 
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
module load prodigal/2.6.3

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load HMMER/3.3.2
module load Graphviz/5.0.0

source .venv/bin/activate

export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH
export HF_HOME=/home/rturnbull/wytamma/huggingface
export TORCH_HOME=/data/gpfs/projects/punim2199/torch-hub

CHECKPOINT=/data/gpfs/projects/punim2199/rob/barbet/logs/bac120-stack-esm6.td-b4f768l2g2lr0.0002ep70a512s32prod-r226/version_0/checkpoints/weights-epoch\=69-valid_loss\=0.29.ckpt
barbet \
    --checkpoint $CHECKPOINT \
    --output-dir outputs \
    --input /data/gpfs/projects/punim2199/wytamma/dev/gambit/GCF_000145595.1.fna

CHECKPOINT=/data/gpfs/projects/punim2199/rob/barbet/logs/bac120-stack-esm6.td-b4f1536l2g2lr0.0002ep70a512s32prod-r226/version_0/checkpoints/weights-epoch\=69-valid_loss\=0.17.ckpt
barbet \
    --checkpoint $CHECKPOINT \
    --output-dir outputs \
    --input /data/gpfs/projects/punim2199/wytamma/dev/gambit/GCF_000145595.1.fna

