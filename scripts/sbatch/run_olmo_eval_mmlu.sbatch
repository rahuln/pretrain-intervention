#!/bin/bash

#SBATCH --job-name=evalmmlu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --array=0-56
#SBATCH --output=slurm/evalmmlu-%A-%a.log
#SBATCH --error=slurm/evalmmlu-%A-%a.log
#SBATCH --export=all

# script to run olmo_eval code on model checkpoint for all MMLU subjects

echo "START TIME: $(date)"

# get arguments from command-line
EVAL_DIR=$1          # name of first subdirectory
MODEL_NAME=$2        # path to saved checkpoint
MODEL_DIRNAME=$3     # name of model directory
REVISION_NAME=$4     # name of model revision/step directory

# list of 57 MMLU subjects
subjects=(
    "abstract_algebra"
    "anatomy"
    "astronomy"
    "business_ethics"
    "clinical_knowledge"
    "college_biology"
    "college_chemistry"
    "college_computer_science"
    "college_mathematics"
    "college_medicine"
    "college_physics"
    "computer_security"
    "conceptual_physics"
    "econometrics"
    "electrical_engineering"
    "elementary_mathematics"
    "formal_logic"
    "global_facts"
    "high_school_biology"
    "high_school_chemistry"
    "high_school_computer_science"
    "high_school_european_history"
    "high_school_geography"
    "high_school_government_and_politics"
    "high_school_macroeconomics"
    "high_school_mathematics"
    "high_school_microeconomics"
    "high_school_physics"
    "high_school_psychology"
    "high_school_statistics"
    "high_school_us_history"
    "high_school_world_history"
    "human_aging"
    "human_sexuality"
    "international_law"
    "jurisprudence"
    "logical_fallacies"
    "machine_learning"
    "management"
    "marketing"
    "medical_genetics"
    "miscellaneous"
    "moral_disputes"
    "moral_scenarios"
    "nutrition"
    "philosophy"
    "prehistory"
    "professional_accounting"
    "professional_law"
    "professional_medicine"
    "professional_psychology"
    "public_relations"
    "security_studies"
    "sociology"
    "us_foreign_policy"
    "virology"
    "world_religions"
)

# get subject
SUBJECT=${subjects[$SLURM_ARRAY_TASK_ID]}
DATASET=mmlu_std_${SUBJECT}

# construct output directory
OUTDIR=results/${EVAL_DIR}/mmlu_std_0shot/${SUBJECT}/${MODEL_DIRNAME}/${REVISION_NAME}
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
    --split test \
    --num-shots 0 \
    --batch-size 8 \
    --full-output-file ${OUTDIR}/predictions.jsonl \
    --metrics-file ${OUTDIR}/metrics.json \
    --model-max-length 4096 \
    --max-batch-tokens 40960 \
    --num-recorded-inputs 3

echo "END TIME: $(date)"
