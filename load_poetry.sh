#!/bin/bash
# Load the environment variables
module purge
module load GCCcore/11.3.0
module load Python/3.10.4
export PATH=/data/gpfs/projects/punim2199/poetry-env/bin:$PATH