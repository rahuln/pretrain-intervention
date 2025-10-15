""" script to post-process correctness label results and output as a single
    matrix with stacked labels for all instances across multiple tasks """

from argparse import ArgumentParser
from glob import glob
import json
import math
import os
import sys

import numpy as np
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--eval_dir", type=str, required=True,
                    help="name of eval subdirectory in 'results' directory, "
                         "e.g., 'eval' if results are in 'results/eval'")
parser.add_argument("--model_name", type=str, required=True,
                    help="name of model or model directory")
parser.add_argument("--dataset_group", type=str, required=True,
                    help="which group of datasets to collect results for; "
                         "this should be the same name as the directory "
                         "for the dataset, e.g., 'mmlu_std_0shot' if the "
                         "results are in 'results/eval/mmlu_std_0shot'")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to .npz file to save label")
parser.add_argument("--resdir_fmt", type=str,
                    default="stage1-step{step}-tokens{num_tokens}B",
                    help="format string for final results subdirectory")
parser.add_argument("--pred_basename", type=str, default="predictions",
                    help="base name of predictions JSONL file")
parser.add_argument("--olmes_fmt", action="store_true",
                    help="predictions stored in olmes format (not olmo_eval)")
parser.add_argument("--metric", type=str, default="acc_raw",
                    help="metric that was used for calculating correctness")
parser.add_argument("--start_step", type=int, default=1000,
                    help="starting pretraining step")
parser.add_argument("--end_step", type=int, default=30000,
                    help="ending pretraining step")
parser.add_argument("--step_gap", type=int, default=1000,
                    help="gap between consecutive steps")
parser.add_argument("--num_runs", type=int, default=1,
                    help="number of repeated runs")


def load_correctness_labels(predictions, metric, olmes_fmt=False):
    """ load all correctness labels for given metric """
    if olmes_fmt:
        if isinstance(predictions, str):
            with open(predictions, "r") as f:
                predictions = [json.loads(line) for line in f.readlines()]
        labels = np.array([elem["metrics"][metric] for elem in predictions])
    else:
        if isinstance(predictions, str):
            with open(predictions, "r") as f:
                predictions = json.load(f)
        labels = np.array([elem["prediction"]["metrics"][metric]
                           for elem in predictions["instance_predictions"]])
    return labels


def main(args):
    """ main function """

    # check if output file exists, make directory for it
    if os.path.exists(args.outfile):
        print("result already exists, exiting...")
        sys.exit()
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)

    # list of ParaRel relations and representative template index for each
    relations = ["p17", "p27", "p36", "p127", "p131", "p138", "p176", "p178",
                 "p276", "p495", "p1376", "p1412"]
    template_index = {
        "p17": 0,
        "p27": 2,
        "p36": 0,
        "p127": 0,
        "p131": 0,
        "p138": 1,
        "p176": 0,
        "p178": 0,
        "p276": 1,
        "p279": 0,
        "p495": 2,
        "p1376": 0,
        "p1412": 0,
    }

    # list of MMLU subjects
    mmlu_subjects = ["abstract_algebra", "anatomy", "astronomy",
                     "business_ethics", "clinical_knowledge",
                     "college_biology", "college_chemistry",
                     "college_computer_science", "college_mathematics",
                     "college_medicine", "college_physics",
                     "computer_security", "conceptual_physics", "econometrics",
                     "electrical_engineering", "elementary_mathematics",
                     "formal_logic", "global_facts", "high_school_biology",
                     "high_school_chemistry", "high_school_computer_science",
                     "high_school_european_history", "high_school_geography",
                     "high_school_government_and_politics",
                     "high_school_macroeconomics", "high_school_mathematics",
                     "high_school_microeconomics", "high_school_physics",
                     "high_school_psychology", "high_school_statistics",
                     "high_school_us_history", "high_school_world_history",
                     "human_aging", "human_sexuality", "international_law",
                     "jurisprudence", "logical_fallacies", "machine_learning",
                     "management", "marketing", "medical_genetics",
                     "miscellaneous", "moral_disputes", "moral_scenarios",
                     "nutrition", "philosophy", "prehistory",
                     "professional_accounting", "professional_law",
                     "professional_medicine", "professional_psychology",
                     "public_relations", "security_studies", "sociology",
                     "us_foreign_policy", "virology", "world_religions"]

    # select dataset names based on group
    if "pararel" in args.dataset_group:
        dataset_names = relations
    elif "mmlu" in args.dataset_group:
        dataset_names = mmlu_subjects
    else:
        dataset_names = [args.dataset_group]


    # load all correctness labels
    steps = np.arange(args.start_step,
                      args.end_step + args.step_gap,
                      args.step_gap)
    num_instances = dict()
    correctness_labels = {run : list() for run in range(1, 1 + args.num_runs)}
    for run in range(1, 1 + args.num_runs):
        for step in tqdm(steps, desc=f"run{run}"):

            # get number of tokens, construct results directory name
            if "7B" in args.model_name:
                num_tokens = math.ceil(4096 * 1024 * step / 1e9)
            elif "1B" in args.model_name:
                num_tokens = math.ceil(4096 * 512 * step / 1e9)
            resdir = args.resdir_fmt.format(step=step, num_tokens=num_tokens,
                                            run=run)

            labels_for_step = list()

            for name in dataset_names:

                # construct whole filename for predictions, get labels
                if "pararel" in args.dataset_group:
                    filename = os.path.join("results", args.eval_dir,
                                            args.dataset_group, name,
                                            f"template{template_index[name]}",
                                            args.model_name.replace("/", "-"),
                                            resdir,
                                            f"{args.pred_basename}.jsonl")
                elif "mmlu" in args.dataset_group:
                    filename = os.path.join("results", args.eval_dir,
                                            args.dataset_group, name,
                                            args.model_name.replace("/", "-"),
                                            resdir,
                                            f"{args.pred_basename}.jsonl")
                else:
                    filename = os.path.join("results", args.eval_dir,
                                            args.dataset_group,
                                            args.model_name.replace("/", "-"),
                                            resdir,
                                            f"{args.pred_basename}.jsonl")
                labels = load_correctness_labels(filename, args.metric,
                                                 olmes_fmt=args.olmes_fmt)
                labels_for_step.extend(labels)
                num_instances[name] = len(labels)

            correctness_labels[run].append(labels_for_step)

    # convert list of lists of labels to numpy arrays, re-label keys if there
    # is only a single run
    correctness_labels = {f"run{k}" : np.array(v).T
                          for k, v in correctness_labels.items()}
    if len(correctness_labels) == 1:
        correctness_labels = {"labels" : correctness_labels["run1"]}

    # build list of dataset labels for all instances
    dataset_labels = list()
    for name in dataset_names:
        dataset_labels.extend([name] * num_instances[name])

    np.savez(args.outfile,
             steps=steps,
             dataset_labels=dataset_labels,
             **correctness_labels)


if __name__ == "__main__":
    main(parser.parse_args())
