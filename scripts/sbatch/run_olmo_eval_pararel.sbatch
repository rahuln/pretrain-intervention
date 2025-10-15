#!/bin/bash

#SBATCH --job-name=evalpara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm/evalpara-%A-%a.log
#SBATCH --error=slurm/evalpara-%A-%a.log
#SBATCH --export=all

# script to run olmo_eval code on model checkpoint for 12 ParaRel relations

echo "START TIME: $(date)"

# get arguments from command-line
EVAL_DIR=$1          # name of first subdirectory
MODEL_NAME=$2        # path to saved checkpoint
MODEL_DIRNAME=$3     # name of model directory
REVISION_NAME=$4     # name of model revision/step directory

# list of 12 ParaRel relations and their representative template indices
relations=(
    "p17"
    "p27"
    "p36"
    "p127"
    "p131"
    "p138"
    "p176"
    "p178"
    "p276"
    "p495"
    "p1376"
    "p1412"
)
declare -A template_index=(
    ["p17"]=0
    ["p27"]=2
    ["p36"]=0
    ["p127"]=0
    ["p131"]=0
    ["p138"]=1
    ["p176"]=0
    ["p178"]=0
    ["p276"]=1
    ["p495"]=2
    ["p1376"]=0
    ["p1412"]=0
)

# get template index
REL=${relations[$SLURM_ARRAY_TASK_ID]}
TEMPL_IDX=${template_index[$REL]}
DATASET=pararel_patterns_std_${REL}_temp${TEMPL_IDX}

# construct output directory
OUTDIR=results/${EVAL_DIR}/pararel_std_0shot/${REL}/template${TEMPL_IDX}/${MODEL_DIRNAME}/${REVISION_NAME}
mkdir -p ${OUTDIR}

# check for existing results
if [ -f ${OUTDIR}/metrics.json ]
then
    echo "results already exist, exiting..."
    exit
fi

set -x

MKL_SERVICE_FORCE_INTEL=GNU python -m olmo_eval.run_lm_eval \
    --model lm::pretrained=${MODEL_NAME} \
    --task ${DATASET} \
    --split train \
    --num-shots 0 \
    --batch-size 8 \
    --full-output-file ${OUTDIR}/predictions.jsonl \
    --metrics-file ${OUTDIR}/metrics.json \
    --model-max-length 4096 \
    --max-batch-tokens 40960 \
    --num-recorded-inputs 3 \
    --limit 160000

echo "END TIME: $(date)"
