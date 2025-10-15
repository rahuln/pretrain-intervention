#!/bin/bash

#SBATCH --job-name=decode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1G
#SBATCH --time=3:00:00
#SBATCH --array=0-499
#SBATCH --output=slurm/decode-%A-%a.log
#SBATCH --error=slurm/decode-%A-%a.log
#SBATCH --export=all


DATA_ORDER_FILE_PATH=$1
TRAIN_CONFIG_PATH=$2
DATADIR=$3

START_BATCH_IDX=$(( $SLURM_ARRAY_TASK_ID * 10 ))
END_BATCH_IDX=$(( $SLURM_ARRAY_TASK_ID * 10 + 10 ))

BATCH=$(( $START_BATCH_IDX / 1000 + 1 ))
SUBDIR=$(printf "%03d" $BATCH)

echo "start ${START_BATCH_IDX}, end ${END_BATCH_IDX}, subdir ${SUBDIR}"

time python -m scripts.decode_batches \
    --data_order_file_path ${DATA_ORDER_FILE_PATH} \
    --train_config_path ${TRAIN_CONFIG_PATH} \
    --start_batch_idx $START_BATCH_IDX \
    --end_batch_idx $END_BATCH_IDX \
    --output_dir ${DATADIR}/raw_text/${SUBDIR} \
    --tokenizer_path allenai/OLMo-2-0425-1B
