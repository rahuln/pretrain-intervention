#!/bin/bash

#SBATCH --job-name=download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=3:00:00
#SBATCH --array=1-1117
#SBATCH --output=slurm/download-%A-%a.log
#SBATCH --error=slurm/download-%A-%a.log
#SBATCH --export=all

echo "START: $(date +'%Y-%m-%d %H:%M:%S')"

DATA_FILES=$1
DATADIR=$2

URL=$(sed -n ${SLURM_ARRAY_TASK_ID}p ${DATA_FILES})
FILENAME=$(echo ${URL} | cut -d'/' -f 4-)
DIRNAME=${DATADIR}/$(dirname ${FILENAME})

mkdir -p ${DIRNAME}
time wget $URL -P ${DIRNAME}

echo "END: $(date +'%Y-%m-%d %H:%M:%S')"
