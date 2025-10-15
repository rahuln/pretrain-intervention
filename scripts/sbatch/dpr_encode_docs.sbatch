#!/bin/bash

#SBATCH --job-name=encdoc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --array=0-499
#SBATCH --output=slurm/encdoc-%A-%a.log
#SBATCH --error=slurm/encdoc-%A-%a.log
#SBATCH --export=all

echo "START: $(date)"

DATADIR=$1
CKPT=$2
OUTDIR=$3
INDEX=$(( $SLURM_ARRAY_TASK_ID + 1 ))

FNAME=$(ls -1 ${DATADIR} | sort | sed -n ${INDEX}p)
FILENUM=$(basename $FNAME | sed -z "s/.json.gz//g")

mkdir -p ${OUTDIR}/${CKPT}

time python -m scripts.dpr_encode_docs \
    --model_name facebook/dpr-ctx_encoder-multiset-base \
    --input_file ${DATADIR}/${FNAME} \
    --output_file ${OUTDIR}/${CKPT}/${FILENUM}.npy \
    --batch_size 64 \
    --max_length 512

echo "END: $(date)"
