""" Given a dataset, a subset of instances indices, a path to a set of JSON
    files with documents, and a path to a scores file with instance-document
    similarity scores, find and save the top-k highest-scoring documents for
    each instance, for some value of k """

from argparse import ArgumentParser
import gzip
import json
import os
import sys

from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
import scipy as sp
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help="name of dataset")
parser.add_argument("--indices_file", type=str, required=True,
                    help="path to file with indices of instances")
parser.add_argument("--doc_counts_file", type=str, required=True,
                    help="path to array with doc counts per JSON file")
parser.add_argument("--path_to_doc_files", type=str, required=True,
                    help="path to JSON files with document text")
parser.add_argument("--scores_file", type=str, required=True,
                    help="path to instance-doc scores file (sparse matrix)")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to output file to save results")
parser.add_argument("--data_config", type=str, default=None,
                    help="HuggingFace dataset config")
parser.add_argument("--indices_file_key", type=str, default="orig_inst_idxs",
                    help="key of instance indices in indices file")
parser.add_argument("--select_indices", action="store_true",
                    help="select instance indices from rows of scores matrix")
parser.add_argument("--subset", type=str, nargs="+", default=["validation"],
                    help="subset(s) of the dataset to use")
parser.add_argument("--question_col", type=str, default="question",
                    help="name of column containing question/query")
parser.add_argument("--answer_col", type=str, default="answer",
                    help="name of column containing answer")
parser.add_argument("--datafmt", type=str,
                    default="data/huggingface/pararel_patterns_{rel}",
                    help="format string for path to single ParaRel dataset")
parser.add_argument("--file_num_offset", type=int, default=0,
                    help="offset to apply to file number in JSON file name")
parser.add_argument("--doc_idx_offset", type=int, default=0,
                    help="offset to apply to document index")
parser.add_argument("--scores_dense", action="store_true",
                    help="scores matrix is not sparse")
parser.add_argument("--min_score", type=int, default=None,
                    help="exclude documents with scores equal to or less than "
                         "the given minimum score")
parser.add_argument("--k", type=int, default=10,
                    help="number of highest-scoring documents per instance")


def get_document_by_index(directory, doc_counts, index, offset=0):
    """ Given a directory of .json.gz files, an array of document counts per
        file, and a global index, return the document at that index"""

    # determine which file the index falls into
    cumulative = 0
    for file_num, count in enumerate(doc_counts):
        if index < cumulative + count:
            local_index = index - cumulative
            filename = os.path.join(directory, f"{file_num + offset}.json.gz")
            break
        cumulative += count
    else:
        raise IndexError("Index out of range of total documents.")

    # read that file and extract the specific line
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == local_index:
                data = json.loads(line)
                return data["text"]

    raise RuntimeError("Failed to read the expected document.")


def main(args):
    """ main function """

    # check for existing result
    if os.path.exists(args.outfile):
        print("result already exists, exiting...")
        sys.exit()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # load questions and answers from dataset
    questions, answers = list(), list()
    if args.dataset == "pararel":
        relations = ["p17", "p27", "p36", "p127", "p131", "p138", "p176",
                     "p178", "p276", "p495", "p1376", "p1412"]
        for rel in relations:
            dataset = load_from_disk(args.datafmt.format(rel=rel))
            for row in dataset["train"]:
                questions.append(row["question"])
                answers.append(row["object"])
    elif args.dataset == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
        questions = dataset["test"]["question"]
        answers = [elem["choices"][elem["answer"]] for elem in dataset["test"]]
    elif args.dataset == "openbookqa":
        dataset = load_dataset("allenai/openbookqa", "main")
        dataset = concatenate_datasets([dataset["test"],
                                        dataset["validation"],
                                        dataset["train"]])
        questions = dataset["question_stem"]
        answers = list()
        for row in dataset:
            answer_index = row["choices"]["label"].index(row["answerKey"])
            answers.append(row["choices"]["text"][answer_index])
    else:
        # handles all other HuggingFace datasets, with option to concatenate
        # multiple splits of the dataset together
        dataset = load_dataset(args.dataset, args.data_config)
        datasets = [dataset[subset] for subset in args.subset]
        dataset = concatenate_datasets(datasets)
        questions = dataset[args.question_col]
        answers = dataset[args.answer_col]

    # load indices file, get instance indices, subset questions and answers
    doc_idxs = np.load(args.indices_file)
    inst_idxs = doc_idxs[args.indices_file_key]
    questions = [questions[idx] for idx in inst_idxs]
    answers = [answers[idx] for idx in inst_idxs]

    # load sparse matrix of instance-document scores (assume the number of
    # rows matches the number of selected instances)
    if args.scores_dense:
        scores_matrix = np.load(args.scores_file)
    else:
        scores_matrix = sp.sparse.load_npz(args.scores_file).tocsr()
    if args.select_indices:
        scores_matrix = scores_matrix[inst_idxs, :]
    assert scores_matrix.shape[0] == len(questions), "mismatched size"

    # load document counts file
    doc_counts = np.load(args.doc_counts_file)

    # for each instance, get the top-k highest-scoring documents
    results = list()
    for i, idx in enumerate(tqdm(inst_idxs, ncols=100,
                                 ascii=True, desc="getting docs")):
        question, answer = questions[i], answers[i]

        # get indices of documents sorted by scores
        if args.scores_dense:
            scores = scores_matrix[i]
        else:
            scores = scores_matrix[i].toarray().flatten()
        doc_idxs_sorted = np.argsort(scores)[::-1]

        for rank, doc_idx in enumerate(doc_idxs_sorted[:args.k]):

            # skip docs with score less than min score, if specified
            score = float(scores[doc_idx])
            if args.min_score is not None and score <= args.min_score:
                continue

            document = get_document_by_index(args.path_to_doc_files,
                                             doc_counts,
                                             doc_idx + args.doc_idx_offset,
                                             offset=args.file_num_offset)

            results.append({
                "index" : int(idx),
                "question" : question,
                "answer" : answer,
                "document" : document,
                "rank" : int(rank + 1),
                "score" : score,
            })

    # save results to file
    final_results = {"results" : results, "args" : vars(args)}
    with open(args.outfile, "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main(parser.parse_args())
