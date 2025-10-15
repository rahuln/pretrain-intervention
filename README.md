# Interventional Analyses on Pretraining Data

This repository contains the code to run the experiments described in:

**Rewriting History: A Recipe for Interventional Analyses to Study Data Effects on Model Behavior**<br>
Rahul Nadkarni, Hila Gonen, Yanai Elazar, Noah A. Smith<br>
_arXiv preprint, 2025_

## Setting up the code

### Environment

To set up a `conda` environment for this repository, run the following commands:

```bash
conda create -n pretrain-intervention python=3.10
conda activate pretrain-intervention
pip install -r requirements.txt
```

### Additional repositories

In addition to this repository, you need to set up the following:

* The [What's In My Big Data (WIMBD)](https://github.com/allenai/wimbd) tool for getting term counts
* The `pretrain-intervention` branch of [this fork of the OLMo Github repo](https://github.com/rahuln/OLMo/tree/mod-pretrain) for re-running pretraining
* The [OLMES repo](https://github.com/allenai/olmes) for running evaluations

Please follow the instructions in each repository for proper installation, including creating separate `conda` environments for each if desired.


## Setting up the pretraining data

### Downloading the files

You can download the OLMo 2 pretraining data by getting a list of `.npy` files from an OLMo 2 pretraining config file and then downloading them with a provided Slurm script. To get a list of the `.npy` files, run the following commands:

```bash
wget https://raw.githubusercontent.com/allenai/OLMo/refs/heads/main/configs/official-0425/OLMo2-1B-stage1.yaml
grep "http://olmo-data.org/preprocessed" OLMo2-1B-stage1.yaml | sort | uniq | cut -d" " -f 6 > olmo_data_files.txt
```

Then you can run the following Slurm script to download all the files in parallel to a specified directory `$DATADIR`:

```bash
sbatch scripts/sbatch/download_pretraining_data_files.sbatch olmo_data_files.txt ${DATADIR}
```

### Decoding into batches

In order to convert the downloaded data files into files that contain the ordered data (either raw text documents or batches of token IDs) seen during OLMo 2 pretraining, you first need to construct the data ordering file using the `OLMo` pretraining code. First, go to the cloned `OLMo` Github repo and update the paths to the data files in `configs/official-0425/OLMo2-1B-stage1.yaml` to point to the data files you downloaded in the previous step. Then, assuming you are using a machine with 4 GPUs, from the `OLMo` Github repo directory you can run the following command:

```bash
torchrun --nproc_per_node=4 -m scripts.train configs/official-0425/OLMo2-1B-stage1-yaml --dry_run
```

This will create the data ordering indices file as a memory-mapped array at `results/OLMo2-1B-stage/train_data/global_indices.npy` (you can perform similar steps to get the indices file for OLMo 2 7B). Given this indices file, the pretraining config file, and the path to the directory of downloaded data files, you can use the Python script at `scripts/decode_batches.py` to decode the data files into `.json.gz` files with the decoded raw text documents ordered as they were seen during pretraining. You can use our provided Slurm script to do this for multiple data batches (set up to decode documents from the first 5,000 steps of pretraining by default):

```bash
sbatch scripts/sbatch/decode_batches.sbatch ${DATA_ORDER_FILE_PATH} ${TRAIN_CONFIG_PATH} ${DATADIR}
```

this will create subdirectories in `${DATADIR}` for each 1,000 batches/files (e.g., `${DATADIR}/001` for the first 1,000 batches/files).

If you want to save `.npy` files of token IDs rather than the raw text of the documents, you can add the `--keep_tokenized` flag when running `scripts/decode_batches.py` in the Slurm script. Having `.npy` files of token IDs will be necessary later for modifying the data batches to perform interventions.

### Building document info files

In order to modify data batches to perform an intervention in later steps, you will also need document info files. These are NumPy arrays that contain the batch index, sequence index, and starting and ending token positions (not including the EOS token) of each document in a subdirectory as created by the previous step when using the `--keep_tokenized` flag. To build these files, you will first need to compile the token convolution function code in `scripts/convolution.c` by using the following command:

```bash
gcc -shared -o scripts/convolution.so -fPIC scripts/convolution.c
```

Then use the script `scripts/get_doc_info_from_batch.py` to create document info files for each data batch (e.g., set of 1,000 batches of tokens).

## Evaluation

### Running evaluations

Evaluating model checkpoints on datasets is a core part of the analysis, both for identifying a set of target evaluation items for our intervention as well as to measure the effect of the intervention. We use the code in the `olmo_eval/` directory to evaluate models on ParaRel and MMLU, and the `olmes` command-line utility for evaluating on OpenBookQA and SciQ.

To evaluate model checkpoints on ParaRel or MMLU, see the Slurm scripts at `scripts/sbatch/run_olmo_eval_pararel.sbatch` and `scripts/sbatch/run_olmo_eval_mmlu.sbatch`, respectively. These scripts are set up to run evaluations for a specified HuggingFace model and revision or a local model checkpoint, parallelizing over all 12 ParaRel relations that we use or all 57 MMLU subjects. The `$MODEL_NAME` argument accepts both a path to a local model checkpoint as well as a HuggingFace model and revision, e.g., `allenai/OLMo-2-0425-1B,revision=stage1-step10000-tokens21B,trust_remote_code=True`.

For OpenBookQA and SciQ, we evaluate on the `olmes` tasks `openbookqa:rc::xlarge` and `sciq:rc::xlarge`, respectively. For example, to run evaluation on SciQ for the HuggingFace revision of OLMo 2 1B at step 10000, you can use the command:

```bash
olmes --model olmo-2-1b-0425 \
    --model-args '{"trust_remote_code": true}' \
    --revision stage1-step10000-tokens21B \
    --task sciq:rc::xlarge \
    --limit 15000 \
    --num-shots 0 \
    --output-dir results/eval/sciq_std_0shot/allenai-OLMo-2-0425-1B/stage1-step10000-tokens21B
```

To run evaluation on SciQ for a local model checkpoint, you can use the command:

```bash
olmes --model olmo-2-1b-0425 \
    --model-args '{"model_path": "/path/to/model_name/checkpoint_name"}' \
    --task sciq:rc::xlarge \
    --limiit 15000 \
    --num-shots 0 \
    --output-dir results/eval_modified/sciq_std_0shot/model_name/checkpoint_name
```

### Collecting evaluation results

We include `scripts/get_correctness_labels.py` to concatenate a set of evaluation results across multiple steps or revisions of a model or across multiple repeated pretraining runs, saving the correctness labels as a NumPy array in a `.npz` file. This format is used by our Jupyter notebooks that identify which evaluation items to intervene on. The script expects results to be saved in a particular directory structure; see the script for details.

## Performing interventions on data batches

### ParaRel

To perform intervention experiments using ParaRel, we follow these steps:

1.  We use the `wimbd search` command to search for subject and object names from ParaRel in order to get co-occurrence counts. The list of search terms for each ParaRel relation that we use can be found in the text files under `data/search_terms`. To get counts for each search term for a particular relation, use the Slurm script we provided as follows:
    ```bash
    REL=p178
    NUM_TERMS=$(wc -l data/search_terms/pararel_patterns_${REL}_train.txt)
    sbatch --array=1-${NUM_TERMS} scripts/sbatch/search_for_term_pararel.sbatch ${DIRNAME} ${OUTDIR} ${REL}
    ```
    Where `${DIRNAME}` is the path to the enclosing directory for the `.json.gz` files from a particular data batch and `${OUTDIR}` is the path to the desired output directory. This script will output a `.json` file for each search term, containing the data file and document(s) (i.e., line number(s)) where each term occurs and the starting and ending character position of each occurrence.
2.  Use `scripts/build_term_doc_matrix.py` and `scripts/build_term_pair_doc_matrix.py` to build sparse matrices of term-document occurrences and term pair-document co-occurrences from the `.json` term search result files generated from the previous step. These will be used in subsequent steps for identifying matching documents in a data batch to perform the intervention.
3.  Use previously-generated evaluation results, the term-document occurrence and term pair-document co-occurrence matrices, and the document info files to construct a _document indices_ file which contains the indices of which documents to swap to perform the intervention. How this file is constructed will depend on the specific experiment (e.g., removing co-occurrences/occurrences to suppress learning, adding co-occurrences to promote learning, etc.). Steps for how to build the document indices files we used in our experiments can be found in the `create-doc-idxs` notebooks for `pararel` in the `notebooks/` directory.

### MCQA datasets

To perform intervention experiments using the MCQA datasets (MMLU, OpenBookQA, and SciQ), we follow these steps:

1.  Construct files which contain indices of the evaluation items that we will target for our intervention (i.e., items learned at particular checkpoints). See the `create-doc-idxs` notebooks for the `mcqa` datasets in the `notebooks/` directory for how to process evaluation results to construct these indices files.
2.  If you are using BM25 scores to match documents to evaluation items, use the script at `scripts/calc_bm25_scores.py` to calculate these scores for the target evaluation items and all documents in a data batch.
3.  If you are using DPR scores to match documents to evaluation items, follow these steps:
    * Use `scripts/dpr_encode_docs.py` to compute and save DPR context embeddings for all documents in a data batch. We provide a Slurm script `scripts/sbatch/dpr_encode_docs.sbatch` to do this in parallel over all `.json.gz` files within the directory for a data batch.
    * Use `scripts/dpr_encode_instances.py` to compute and save DPR query embeddings for all evaluation items in the target group, providing the evaluation item indices file you constructed in the first step.
    * Use `scripts/dpr_compute_question_doc_sim.py` to compute the similarity scores between document and evaluation item embeddings, saving them in a separate set of files for each original file of documents in the data batch.
    * Finally, use `scripts/dpr_concat_question_doc_sim_arrays.py` to concatenate all document-item similarity score arrays into a single matrix, optionally sparsifying to the top-k highest scores per item (this speeds up the next step).
4.  Once you have built similarity score matrices between your target evaluation items and the documents in your target data batch (i.e., the data batch you want to modify by replacing documents) and source data batch (where you will draw replacement documents from), use `scripts/create_doc_idx_mapping_from_scores.py` to create a mapping between source and target documents for swapping between them. The mapping is constructed to maximize the difference in similarity scores between documents while matching token counts. If you wish to construct an intervention that promotes rather than suppresses learning, use the `--swap_direction` and `--shuffle_candidates` flags.

## Creating a modified pretraining data batch

Once you have created either a document indices file (for ParaRel) or a document mapping file (for the MCQA datasets), use the script at `scripts/swap_docs_between_batches.py` to perform the intervention by swapping documents to build a new set of modified data files. You will need to provide the script with either the document indices file or the document mapping file, as well as the appropriate document info files and paths to the directories of `.npy` files of token IDs corresponding to the data batch that you wish to modify. If you are performing an intervention that attempts to promote learning rather than suppress learning, you should specify the `--skip_extra_docs`, `--pair_by_new_first`, and `--seed` flags when running this script.

Once you have created the modified data batch as a set of `.npy` files, use the script at `scripts/convert_arr_to_memmap.py` to convert them into files containing memory-mapped arrays. This is the data format expected by the OLMo pretraining code.

## Re-running pretraining

Finally, you can use the OLMo pretraining code to re-run pretraining over the modified data batch you have constructed. To do so, you should first construct a modified config file to set up your experiment. You can model this after existing config files in the OLMo Github repo (e.g., `configs/official-0425/OLMo2-1B-stage1.yaml` for OLMo 2 1B). Key settings you should change include:

* `run_name`: Set this to the name of your experiment.
* `load_path`: The path to the downloaded files for the OLMo checkpoint you wish to initialize from. Links to these checkpoints can be found in the `.csv` files in the `configs/` directory.
* `stop_after`: Set this to the number of steps/batches that you want to repeat training over.
* `device_train_microbatch_size`: Set this according to the GPU setup you have to minimize training time while avoiding GPU out-of-memory errors.
* `max_duration`: For the repeated pretraining to work as intended, this must be set to either a total number of tokens or steps (not epochs) for the learning rate schedule to be set properly. This should be the same as the total number of tokens or steps used for the full original OLMo pretraining runs, which is `4e12T` tokens for OLMo 2 1B and `928646` steps for OLMo 2 7B.
* Under the `data` section of the config file, adjust the following settings:
    * `shuffle_train`: Set this to `False`. This is a setting that was added to prevent the code from shuffling the training dataloader when it is initialized.
    * `set_start_index`: Set this to `False`. This is a setting that was added to prevent the code from setting the starting index of the training dataset to the step of the initial loaded model checkpoint.
    * `paths`: Set these as the paths to the modified data batch files you created previously, in the same order.

Once you have set up a config file for re-running pretraining, you can run the following training command (assuming a 4-GPU setup):

```bash
torchrun --nproc_per_node=4 -m scripts.train /path/to/config.yaml --run_name=${RUN_NAME}
```

You can specify your own `${RUN_NAME}` for your experiment (for multiple repeated runs, we simply add a run number to the name, e.g., `--run_name=experiment/run1`). Once training is complete, you will need to unshard the resulting checkpoint and then convert it to HuggingFace format to use it with the evaluation code. This can be done with the following commands:

```bash
torchrun --nproc_per_node=4 -m scripts.train \
    /path/to/config.yaml \
    --run_name=${RUN_NAME}-unsharded \
    --load_path=/path/to/sharded/checkpoint \
    --force_save_unsharded \
    --dry_run

python scripts/convert_olmo2_to_hf.py \
    --input_dir /path/to/unsharded/checkpoint \
    --output_dir /path/to/hf/checkpoint \
    --tokenizer_json_path olmo_data/tokenizers/allenai_dolma2.json
```

You can then follow the steps in the "Evaluation" section above with the HuggingFace format model checkpoint.

