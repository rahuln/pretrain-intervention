""" build a sparse matrix of {term pair, document} co-occurrences where each
    co-occurrence ensures that there is no overlap in the text spans of the
    search terms in each pair
"""

from argparse import ArgumentParser
from glob import glob
import json
import os
import sys

from datasets import load_dataset, load_from_disk
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--matches_dir", type=str, required=True,
                    help="path to directory with files containing match "
                         "locations for each search term")
parser.add_argument("--search_terms_file", type=str, required=True,
                    help="path to file with list of search terms")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="path to ParaRel Patterns dataset to load")
parser.add_argument("--doc_info_file", type=str, required=True,
                    help="path to doc info file for this data batch")
parser.add_argument("--offset", type=int, required=True,
                    help="offset for initial file number")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to output file")


def get_term_pair_doc_occurrences_no_overlap(term1_dict, term2_dict,
    doc_id_to_index):
    """
    Given two dictionaries of occurrences of a search term within a corpus
    (including starting and ending character locations of each occurrence) and
    a mapping from document ID to document index, get a list of indices of
    documents where the two terms co-occur, ensuring that their text spans do
    not overlap """

    def contains(lims1, lims2):
        """ checks if one character window entirely contains the other """
        if lims1[0] >= lims2[0] and lims1[1] <= lims2[1]:
            return True
        elif lims2[0] >= lims1[0] and lims2[1] <= lims1[1]:
            return True
        return False

    # keep track of document indices
    doc_idxs = set()

    # iterate through files where 1st term occurs
    for file_path, term1_occurrences in term1_dict.items():
        file_num = int(os.path.basename(file_path).replace(".json.gz", ""))

        # check if 2nd term occurs at all in this file
        if file_path in term2_dict:
            term2_occurrences = term2_dict[file_path]

            # iterate through occurrences of 1st term
            for term1_occurrence in term1_occurrences:
                term1_line = term1_occurrence['line_num']
                term1_submatches = term1_occurrence['submatches']

                # check if document is already in set, for speed-up
                doc_id = f"{file_num}-{term1_line}"
                if doc_id_to_index[doc_id] in doc_idxs:
                    continue

                # iterate through occurrences of 2nd term
                for term2_occurrence in term2_occurrences:
                    term2_line = term2_occurrence['line_num']
                    term2_submatches = term2_occurrence['submatches']

                    # check if terms occur on the same line (i.e., within
                    # the same document)
                    if term1_line == term2_line:
                        for term1_match in term1_submatches:
                            term1_lims = (term1_match['start_col'],
                                            term1_match['end_col'])

                            for term2_match in term2_submatches:
                                term2_lims = (term2_match['start_col'],
                                                term2_match['end_col'])

                                # check if one text span contains the other
                                if not contains(term1_lims, term2_lims):
                                    doc_idxs.add(doc_id_to_index[doc_id])

    # ensure no repeats in set of indices
    doc_idxs = sorted(set(doc_idxs))

    return doc_idxs


def main(args):
    """ main function """

    # make output directory (if it does not exist), check for output file
    if not args.outfile.endswith(".npz"):
        args.outfile = args.outfile + ".npz"
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    if os.path.exists(args.outfile):
        print("results already exist, exiting...")
        sys.exit()

    # load list of search terms, built term-to-index mapping
    with open(args.search_terms_file, "r") as f:
        term_list = [line.strip() for line in f.readlines()]
    term_to_idx = {term : i for i, term in enumerate(term_list)}

    # load all dictionaries with matches
    files = glob(os.path.join(args.matches_dir, "*.json"))
    dicts = dict()
    for fname in tqdm(files, ascii=True, desc="loading matches dicts"):
        with open(fname, "r") as f:
            results = json.load(f)
        idx = int(os.path.basename(fname).replace(".json", "")) - 1
        dicts[idx] = results["matches"]

    # get file lengths (number of documents in each file) from doc info
    doc_info = np.load(args.doc_info_file)
    file_lengths = np.bincount(doc_info[:, 0])

    # set up document IDs using file lengths and offset
    document_ids = set()
    for idx, num_docs in enumerate(file_lengths):
        for doc_num in range(num_docs):
            document_ids.add(f"{idx+args.offset}-{doc_num+1}")

    # convert document IDs set to a sorted list, create doc-to-index mapping
    key = lambda doc_id: (int(doc_id.split("-")[0]), int(doc_id.split("-")[1]))
    document_ids = sorted(document_ids, key=key)
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(document_ids)}

    # load search term pairs from dataset
    dataset = load_from_disk(args.dataset_path)
    subjects = list(map(str.strip, dataset["train"]["subject"]))
    objects = list(map(str.strip, dataset["train"]["object"]))

    data, row, col = list(), list(), list()
    for r, (subj, obj) in enumerate(tqdm(list(zip(subjects, objects)),
                                    ascii=True,
                                    desc="getting document indices")):
        subj_idx = term_to_idx[subj]
        obj_idx = term_to_idx[obj]
        idxs = get_term_pair_doc_occurrences_no_overlap(dicts[subj_idx],
                                                        dicts[obj_idx],
                                                        doc_id_to_index)

        data.extend([1] * len(idxs))
        row.extend([r] * len(idxs))
        col.extend(idxs)

    # build term pair / document co-occurrence matrix, save to file
    shape = (len(subjects), np.sum(file_lengths))
    term_pair_doc_matrix = coo_matrix((data, (row, col)), shape=shape)
    term_pair_doc_matrix = term_pair_doc_matrix.tocsr()
    sp.sparse.save_npz(args.outfile, term_pair_doc_matrix)


if __name__ == "__main__":
    main(parser.parse_args())

