""" construct a mapping from target to source document index for swapping
    documents, using a set of instance-document scores to match target
    documents to source documents while maximizing decrease in score """

from argparse import ArgumentParser
from collections import defaultdict
import os
import pickle
from pprint import pprint
import random
import sys

import numpy as np
import scipy as sp
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--doc_idxs_file", type=str, required=True,
                    help="path to doc indices file")
parser.add_argument("--target_scores", type=str, required=True,
                    help="path to file with scores for target documents")
parser.add_argument("--source_scores", type=str, required=True,
                    help="path to file with scores for source documents")
parser.add_argument("--target_doc_info", type=str, required=True,
                    help="path to doc info file for target docs")
parser.add_argument("--source_doc_info", type=str, required=True,
                    help="path to doc info file for source docs")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to output file")
parser.add_argument("--scores_sparse", action="store_true",
                    help="scores files are scipy.sparse matrices")
parser.add_argument("--coocc_matrix_fmt", type=str, default=None,
                    help="filename pattern for {term pair, doc} occurrences "
                         "matrix, if using co-occurrences (must be a format "
                         "string which contains '{rel}' for the ParaRel "
                         "relation)")
parser.add_argument("--target_start_batch", type=int, default=500,
                    help="batch ID for first target doc (inclusive)")
parser.add_argument("--target_end_batch", type=int, default=1000,
                    help="batch ID for last target doc (exclusive)")
parser.add_argument("--source_start_batch", type=int, default=0,
                    help="batch ID for first source doc (inclusive)")
parser.add_argument("--source_end_batch", type=int, default=500,
                    help="batch ID for last source doc (exclusive)")
parser.add_argument("--per_inst_doc_count", type=int, default=None,
                    help="number of documents to replace per instance")
parser.add_argument("--swap_direction", action="store_true",
                    help="swap direction of source and target document "
                         "indices when saving final mapping file")
parser.add_argument("--shuffle_candidates", action="store_true",
                    help="shuffle candidate source documents when building "
                         "mapping to prevent target documents from matching "
                         "only earlier source documents")
parser.add_argument("--match_total_doc_count", action="store_true",
                    help="match the total number of replaced documents with "
                         "the count from the doc indices file")
parser.add_argument("--offset_shift", type=int, default=10,
                    help="amount to shift offset by when getting number of "
                         "target documents to match those with co-occurrences")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed")


def truncate_doc_info(doc_info, start_batch, end_batch):
    """ truncate doc info file to specified batch number range """
    if start_batch >= 0:
        start_idx = np.where(doc_info[:, 0] == start_batch)[0][0]
        doc_info = doc_info[start_idx:, :]
    if end_batch <= np.max(doc_info[:, 0]):
        end_idx = np.where(doc_info[:, 0] == end_batch)[0][0]
        doc_info = doc_info[:end_idx, :]
    return doc_info, start_idx


def main(args):
    """ main function """

    # check for output file, create output directory
    if os.path.exists(args.outfile):
        print("result already exists, exiting...")
        sys.exit()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # set random seeds for breaking ties when sorting
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ParaRel relations
    relations = ["p17", "p27", "p36", "p127", "p131", "p138", "p176", "p178",
                 "p276", "p495", "p1376", "p1412"]

    # load scores matrices
    if args.scores_sparse:
        target_scores = sp.sparse.csr_matrix(
            sp.sparse.load_npz(args.target_scores))
    else:
        target_scores = np.load(args.target_scores)
    print("target_scores.shape:", target_scores.shape)

    if args.scores_sparse:
        source_scores = sp.sparse.csr_matrix(
            sp.sparse.load_npz(args.source_scores))
    else:
        source_scores = np.load(args.source_scores)
    print("source_scores.shape:", source_scores.shape)

    # load doc info files, truncate
    target_doc_info = np.load(args.target_doc_info)
    print("target_doc_info.shape:", target_doc_info.shape)
    target_doc_info, target_start_idx = \
        truncate_doc_info(target_doc_info,
                          args.target_start_batch,
                          args.target_end_batch)
    print("target_doc_info.shape, truncated:", target_doc_info.shape)
    target_num_tokens = target_doc_info[:, -1] - target_doc_info[:, -2] + 1

    source_doc_info = np.load(args.source_doc_info)
    print("source_doc_info.shape:", source_doc_info.shape)
    source_doc_info, source_start_idx = \
        truncate_doc_info(source_doc_info,
                          args.source_start_batch,
                          args.source_end_batch)
    print("source_doc_info.shape, truncated:", source_doc_info.shape)
    source_num_tokens = source_doc_info[:, -1] - source_doc_info[:, -2] + 1

    # create mapping from token count to doc indices for source docs
    tokens_to_docs = defaultdict(list)
    for doc_idx, num_tokens in zip(np.arange(source_doc_info.shape[0]),
                                             source_num_tokens):
        tokens_to_docs[num_tokens].append(doc_idx)

    # load doc indices file for removing co-occurrences
    doc_idxs = np.load(args.doc_idxs_file)
   
    # set target doc counts
    if args.per_inst_doc_count is not None:
        num_inst = len(doc_idxs["orig_inst_idxs"])
        matching_doc_counts = args.per_inst_doc_count * np.ones(num_inst, dtype=int)
        total_doc_count = args.per_inst_doc_count * num_inst
    elif args.coocc_matrix_fmt is not None:
        # load co-occurrences matrix, get per-instance matching doc counts
        # load co-occ matrix for each relation, stack, and truncate / subset
        matrices = list()
        for rel in relations:
            matrices.append(sp.sparse.load_npz(args.coocc_matrix_fmt.format(rel=rel)))
        coocc_matrix = sp.sparse.vstack(matrices)
        coocc_matrix = coocc_matrix[doc_idxs["orig_inst_idxs"], :][:, start_idx:]
        print("coocc_matrix.shape:", coocc_matrix.shape)

        # get number of matching docs per instance
        matching_doc_counts = coocc_matrix.sign().sum(axis=1).A1.flatten()
        print("matching_doc_counts.shape:", matching_doc_counts.shape)
        print("np.min(matching_doc_counts):", np.min(matching_doc_counts))
        print("np.max(matching_doc_counts):", np.max(matching_doc_counts))
        total_doc_count = np.sum(np.sign(coocc_matrix.sign().sum(axis=0).A1.flatten()))
        print("np.sum(matching_doc_counts):", total_doc_count)
    else:
        raise ValueError("must specify per_inst_doc_count or coocc_matrix_fmt")

    # get target doc indices based on matching doc counts and scores
    offset = 0
    target_doc_idxs = set()
    while len(target_doc_idxs) < total_doc_count: 
        if offset > 0:
            print(f"total of {len(target_doc_idxs)} too low...")
        target_doc_idxs = set()
        for idx, count in enumerate(tqdm(matching_doc_counts,
                                    desc="getting matching doc counts",
                                    ncols=80, ascii=True)):
            if args.scores_sparse:
                scores = target_scores[idx].toarray().flatten()
            else:
                scores = target_scores[idx]
            # use either number of docs with non-zero score or specified
            # count for this instance, whichever is lower
            num_docs_to_use = np.minimum(int(np.sum(np.abs(np.sign(scores)))),
                                         count + offset)
            idxs = np.argsort(scores)[::-1][:num_docs_to_use]
            target_doc_idxs.update(idxs)
        offset += args.offset_shift

        # if not matching total doc count, only need one loop iteration
        if not args.match_total_doc_count:
            break
    target_doc_idxs = np.array(sorted(target_doc_idxs))
    print("len(target_doc_idxs):", len(target_doc_idxs))

    # get matching instance counts, sort target doc indices by count
    if args.scores_sparse:
        inst_counts = target_scores[:, target_doc_idxs].sign().sum(axis=0).A1.flatten()
    else:
        inst_counts = np.sum(np.sign(target_scores[:, target_doc_idxs]), axis=0)
    sorted_idxs = np.argsort(inst_counts)[::-1]

    # create mapping from target doc ID to source doc ID
    mapping = dict()
    for idx in tqdm(sorted_idxs,
                    desc="mapping target to source docs",
                    ncols=80, ascii=True):
        target_doc_idx = target_doc_idxs[idx]
        target_tokens = target_num_tokens[target_doc_idx]
        while len(tokens_to_docs[target_tokens]) == 0:
            target_tokens += 1
        candidates = tokens_to_docs[target_tokens]
        if args.shuffle_candidates:
            random.shuffle(candidates)
        target_scores_idx = target_scores[:, target_doc_idx]
        if args.scores_sparse:
            inst_idxs = np.nonzero(target_scores_idx.sign().toarray().flatten())[0]
            scores = source_scores[inst_idxs, :][:, candidates].mean(axis=0).A1.flatten()
        else:
            inst_idxs = np.nonzero(np.sign(target_scores_idx))[0]
            scores = source_scores[inst_idxs, :][:, candidates].mean(axis=0)
        source_doc_idx = candidates[np.argmin(scores)]
        mapping[target_doc_idx + target_start_idx] = source_doc_idx + source_start_idx
        candidates.remove(source_doc_idx)

    # save {target doc ID, source doc ID} mapping to file
    if args.swap_direction:
        source_doc_idxs = np.array(sorted(mapping.keys()))
        target_doc_idxs = np.array([mapping[idx] for idx in source_doc_idxs])
    else:
        target_doc_idxs = np.array(sorted(mapping.keys()))
        source_doc_idxs = np.array([mapping[idx] for idx in target_doc_idxs])
    np.savez(args.outfile, orig_idxs=target_doc_idxs, new_idxs=source_doc_idxs)


if __name__ == "__main__":
    main(parser.parse_args())

