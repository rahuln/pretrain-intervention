#!/bin/bash

#SBATCH --job-name=search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=3:00:00
#SBATCH --array=1-1000
#SBATCH --output=slurm/search-%A-%a.log
#SBATCH --error=slurm/search-%A-%a.log
#SBATCH --export=all


echo "START TIME: $(date)"

DIRNAME=$1
OUTDIR=$2
REL=$3

PHRASES_FILE=data/search_terms/pararel_patterns_${REL}_train.txt

# check if specific index exceeds the number of search terms
NUM_PHRASES=$(cat ${PHRASES_FILE} | wc -l)
if [[ ${SLURM_ARRAY_TASK_ID} -gt ${NUM_PHRASES} ]]
then
    echo "index is greater than number of search terms, exiting..."
    exit
fi

# get search term for this array job from file, set up output file
PHRASE=$(sed -n ${SLURM_ARRAY_TASK_ID}p ${PHRASES_FILE})
SAVENAME=${OUTDIR}/${SLURM_ARRAY_TASK_ID}.json
DONENAME=${OUTDIR}/${SLURM_ARRAY_TASK_ID}.json.done

mkdir -p ${OUTDIR}

# check if output file already exists
if [ -f ${DONENAME} ]
then
    echo "result already exists, exiting..."
    exit
fi

# run search with `wimbd search` command
time wimbd search -p "${PHRASE}" --with-locations --json ${DIRNAME}/* > ${SAVENAME}
touch ${DONENAME}

echo "END TIME: $(date)"
