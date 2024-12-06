#!/bin/bash
 
# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=preprocess
 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-119%10
 
# set your minimum acceptable walltime=days-hours:minutes:seconds
#SBATCH -t 12:00:00
 
#SBATCH --mem-per-cpu=12G
 
# SBATCH -p cascade
#SBATCH -p gpu-a100,gpu-h100
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

source /data/gpfs/projects/punim2199/rob/bloodhound/.venv/bin/activate

export HF_HOME=/data/gpfs/projects/punim2199/huggingface
export TORCH_HOME=/data/gpfs/projects/punim2199/torch-hub

DOMAIN=bac120
#DOMAIN=ar53
ESM_LAYERS=30

# HACK
PREPROCESSED_DIR=/data/gpfs/projects/punim2199/preprocessed
SEQTREE_NAME=esm${ESM_LAYERS}-reps.st
SEQTREE=${PREPROCESSED_DIR}/${DOMAIN}/esm${ESM_LAYERS}/${SEQTREE_NAME}


#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm12-reps.st-b16f1536l2g2e4lr0.0001/version_0/checkpoints/epoch=11-step=6700944.ckpt
#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm12-reps-d4.st-b16f2048l2g2e4lr0.0001/version_0/checkpoints/epoch=19-step=11168240.ckpt
#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm12-reps-d4.st-b16f512l0g2e4lr0.0001/version_1/checkpoints/checkpoint-epoch=00-valid_loss=9.9.ckpt
#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm12-reps.st-b16f768l2g2e4lr0.0001/version_0/checkpoints/checkpoint-epoch=00-valid_loss=7.1.ckpt
#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm12-reps.st-b16f768l2g2e4lr0.0001ep20/version_0/checkpoints/checkpoint-epoch=07-valid_loss=6.8.ckpt
#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm12-reps.st-b16f768l2g2e4lr0.0001ep20/version_0/checkpoints/checkpoint-epoch=17-valid_loss=6.2.ckpt
#CHECKPOINT=/home/rturnbull/wytamma/rob/bloodhound/logs/bac120-esm30-reps.st-b16f1536l2g2e4lr0.0001ep20-gpu2/version_0/checkpoints/checkpoint-epoch\=17-valid_loss\=5.8.ckpt
#CHECKPOINT=/data/gpfs/projects/punim2199/rob/bloodhound/logs/bac120-esm30-reps.st-b16f2048l2g2e4lr0.0001ep20-gpu2/version_0/checkpoints/checkpoint-epoch=16-valid_loss=5.8.ckpt
CHECKPOINT=/data/gpfs/projects/punim2199/rob/bloodhound/logs/bac120-esm30-reps.st-b16f2048l2g2e4lr0.0002ep20-gpu2/version_0/checkpoints/checkpoint-epoch\=18-valid_loss\=5.7.ckpt
bloodhound \
    --checkpoint $CHECKPOINT \
    --out-dir outputs \
    --sequence /data/gpfs/projects/punim2199/wytamma/dev/gambit/GCF_000145595.1.fna \
    --image-dir outputs/GCF_000145595.1-images-esm30-2048lr2 \
    --output-csv outputs/GCF_000145595.1-images-esm30-2048lr2.csv \
    --output-gene-csv outputs/GCF_000145595.1-images-esm30-2048lr2-gene.csv \
    --output-averaged-csv outputs/GCF_000145595.1-images-esm30-2048lr2-averaged.csv

bloodhound \
    --checkpoint $CHECKPOINT \
    --out-dir outputs-GCA_001027105.1 \
    --sequence /home/rturnbull/wytamma/rob/GCA_001027105.1_ASM102710v1_genomic.fna \
    --image-dir outputs-GCA_001027105.1/GCA_001027105.1-esm30-2048lr2-images \
    --output-csv outputs-GCA_001027105.1/GCA_001027105.1-esm30-2048lr2.csv \
    --output-gene-csv outputs-GCA_001027105.1/GCA_001027105.1-esm30-2048lr2-gene.csv \
    --output-averaged-csv outputs-GCA_001027105.1/GCA_001027105.1-esm30-2048lr2-averaged.csv