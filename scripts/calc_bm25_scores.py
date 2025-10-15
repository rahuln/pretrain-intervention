""" use the bm25s library to calculate and save top-k BM25 scores (and
    associated document IDs) between instances from 12 ParaRel relations and a
    specified corpus of a subset of OLMo pretraining documents """

from argparse import ArgumentParser
from glob import glob
import gzip
import json
import os
import sys

import bm25s
from datasets import load_from_disk, load_dataset, concatenate_datasets
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from tqdm import tqdm


parser = ArgumentParser(description="calculate top BM25 scores and doc IDs")
parser.add_argument("--datadir", type=str, default=None,
                    help="path to directory with .json.gz document files")
parser.add_argument("--dataset", type=str, default="pararel",
                    help="choice of dataset")
parser.add_argument("--data_config", type=str, default=None,
                    help="config for HuggingFace dataset")
parser.add_argument("--subset", type=str, nargs="+", default=["validation"],
                    help="subset(s) of the dataset to use")
parser.add_argument("--question_col", type=str, default="question",
                    help="name of column containing question/query")
parser.add_argument("--answer_col", type=str, default="answer",
                    help="name of column containing answer")
parser.add_argument("--datafmt", type=str,
                    default="data/huggingface/pararel_patterns_{rel}",
                    help="format string for path to single ParaRel dataset")
parser.add_argument("--start_index", type=int, default=None,
                    help="index of first file to use (inclusive)")
parser.add_argument("--end_index", type=int, default=None,
                    help="index of last file to use (inclusive)")
parser.add_argument("--add_answer", action="store_true",
                    help="include answer in search query for each instance")
parser.add_argument("--indices-file", type=str, default=None,
                    help="path to file with indices of instances")
parser.add_argument("--indices-file-key", type=str, default="orig_inst_idxs",
                    help="key of instance indices in indices file")
parser.add_argument("--max-num-docs", type=int, default=None,
                    help="maximum number of documents to load")
parser.add_argument("--k", type=int, default=10000,
                    help="number of top-BM25-scoring documents to retrieve")
parser.add_argument("--path-to-index", type=str, default=None,
                    help="path to saved index of corpus")
parser.add_argument("--savename", type=str, default=None,
                    help="path to save BM25 scores and doc IDs")


def main(args):
    """ main function """

    # check for existing results, make results directory
    if args.savename is not None:
        if os.path.exists(args.savename):
            print("results already exist, exiting...")
            sys.exit()
        os.makedirs(os.path.dirname(args.savename), exist_ok=True)

    # get sorted list of .json.gz document files
    key = lambda x: int(os.path.basename(x).replace(".json.gz", ""))
    files = sorted(glob(os.path.join(args.datadir, "*.json.gz")), key=key)
    args.start_index = 0 if args.start_index is None else args.start_index
    args.end_index = len(files) if args.end_index is None else args.end_index
    files = files[args.start_index : args.end_index]
    print("first file:", files[0])
    print("last file:", files[-1])

    # set maximum number of documents
    if args.max_num_docs is None:
        max_num_docs = 500000
    else:
        max_num_docs = args.max_num_docs

    # iterate through files and load documents
    progbar = tqdm(range(max_num_docs), desc="loading docs", ascii=True,
                   ncols=100)
    doc_count = 0
    corpus = list()
    for fname in files:
        with gzip.open(fname, "rt", encoding="utf-8") as f:
            for line in f:
                corpus.append(json.loads(line)["text"])
                progbar.update(1)
                doc_count += 1
                if doc_count >= max_num_docs:
                    break
        if doc_count >= max_num_docs:
            break

    print("tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
    print()

    # load index if specified, otherwise build it
    if args.path_to_index is not None \
        and os.path.exists(args.path_to_index):
        print("loading corpus index...\n")
        retriever = bm25s.BM25.load(args.path_to_index, load_corpus=True)
    else:
        print("constructing corpus index...\n")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        if args.path_to_index is not None:
            retriever.save(args.path_to_index, corpus=corpus)

    # loading instances for specified dataset (12 ParaRel relations or MMLU)
    queries = list()
    if args.dataset == "pararel":
        relations = ["p17", "p27", "p36", "p127", "p131", "p138", "p176",
                     "p178", "p276", "p495", "p1376", "p1412"]
        for rel in tqdm(relations, desc="loading instances"):
            dataset = load_from_disk(args.datafmt.format(rel=rel))
            for row in dataset["train"]:
                if args.add_answer:
                    queries.append(f"{row['subject']} {row['object']}")
                else:
                    queries.append(row["subject"])
    elif args.dataset == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
        if args.add_answer:
            queries = [f"{elem['question']} {elem['choices'][elem['answer']]}"
                       for elem in dataset["test"]]
        else:
            queries = dataset["test"]["question"]
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
        if args.add_answer:
            queries = [f"{question} {answer}" for question, answer
                       in zip(questions, answers)]
        else:
            queries = questions
    else:
        # handles all other HuggingFace datasets, with option to concatenate
        # multiple splits of the dataset together
        args.data_config = None if args.data_config == "None" else args.data_config
        dataset = load_dataset(args.dataset, args.data_config)
        datasets = [dataset[subset] for subset in args.subset]
        dataset = concatenate_datasets(datasets)
        if args.add_answer:
            queries = [f"{elem[args.question_col]} {elem[args.answer_col]}"
                       for elem in dataset]
        else:
            queries = dataset[args.question_col]

    print("sample queries:")
    for i in np.random.randint(0, len(queries), size=5):
        print(queries[i])

    print("getting doc IDs and BM25 scores for queries...")

    # prevent top-k value from being larger than corpus size
    if args.k > len(corpus):
        args.k = len(corpus)

    # get indices of instances to use
    if args.indices_file is not None:
        indices = np.load(args.indices_file)
        inst_idxs = indices[args.indices_file_key]
    else:
        inst_idxs = np.arange(len(queries))

    # create sparse matrix of BM25 scores for every {query, doc} pair
    row, col, data = [], [], []
    for i, idx in enumerate(tqdm(inst_idxs, ascii=True, ncols=100,
                                 desc="processing queries")):
        query_tokens = bm25s.tokenize(queries[idx])
        doc_ids, scores = retriever.retrieve(query_tokens, k=args.k)
        scores = scores.flatten()
        try:
            doc_ids = doc_ids[0]
        except IndexError:
            import ipdb; ipdb.set_trace()
        nonzero_idxs = np.nonzero(scores)[0]
        row.extend([i] * len(nonzero_idxs))
        col.extend(doc_ids[nonzero_idxs])
        data.extend(scores[nonzero_idxs])

    # convert to COO format
    scores = coo_matrix((data, (row, col)), shape=(len(inst_idxs), len(corpus)))

    print("scores.shape:", scores.shape)
    print("scores sparsity:", scores.nnz / np.product(scores.shape))
    print()

    # save results
    if args.savename is not None:
        sp.sparse.save_npz(args.savename, scores)


if __name__ == "__main__":
    main(parser.parse_args())

