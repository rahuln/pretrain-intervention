""" build a sparse matrix of {search term, document} occurrences """

from argparse import ArgumentParser
import json
from json import JSONDecodeError
import os
import sys

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--matches_dir", type=str, required=True,
                    help="path to directory with files containing match "
                         "locations for each search term")
parser.add_argument("--doc_info_file", type=str, required=True,
                    help="path to doc info file for this data batch")
parser.add_argument("--offset", type=int, required=True,
                    help="offset for initial file number")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to output file")


def build_term_document_matrix(directory_path, file_lengths, offset):
    """ given a directory of JSON files where each file corresponds to a unique
        search term and contains the output of running a `wimbd search` command
        for that term against a corpus, build a count matrix indicating which
        terms appear in which documents and how often """

    # get a list of all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    json_files = sorted(json_files, key=lambda x: int(os.path.splitext(x)[0]))
    file_numbers = [int(os.path.splitext(f)[0]) for f in json_files]

    # set up document IDs
    document_ids = set()
    for idx, num_docs in enumerate(file_lengths):
        for doc_num in range(num_docs):
            document_ids.add(f"{idx+offset}-{doc_num+1}")

    # read each JSON file and collect document IDs
    good_json_files = list()
    for json_file in tqdm(json_files, desc="getting doc IDs"):
        file_path = os.path.join(directory_path, json_file)
        try:
            with open(file_path, "r") as file:
                matches = json.load(file)
            good_json_files.append(json_file)
        except JSONDecodeError:
            continue
        file_doc_ids = list()
        for key, value in matches["matches"].items():
            file_id = os.path.basename(key).replace(".json.gz", "")
            for entry in value:
                file_doc_ids.append(f"{file_id}-{entry['line_num']}")
            document_ids.update(file_doc_ids)

    # convert document IDs set to a sorted list
    key = lambda doc_id: (int(doc_id.split("-")[0]), int(doc_id.split("-")[1]))
    document_ids = sorted(document_ids, key=key)

    # create a mapping of document ID to column index
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(document_ids)}

    # populate the count matrix
    row, col, data = [], [], []
    for row_idx, json_file in enumerate(tqdm(good_json_files,
                                             desc="getting counts")):
        file_path = os.path.join(directory_path, json_file)
        with open(file_path, "r") as file:
            matches = json.load(file)
        for key, value in matches["matches"].items():
            file_id = os.path.basename(key).replace(".json.gz", "")
            for entry in value:
                doc_id = f"{file_id}-{entry['line_num']}"
                count = len(entry["submatches"])
                row.append(row_idx)
                col.append(doc_id_to_index[doc_id])
                data.append(count)

    shape = (len(good_json_files), len(document_ids))
    term_doc_matrix = coo_matrix((data, (row, col)), shape=shape)
    term_doc_matrix = term_doc_matrix.tocsr()

    return term_doc_matrix


def main(args):
    """ main function """

    # make output directory (if it does not exist), check for output file
    if not args.outfile.endswith(".npz"):
        args.outfile = args.outfile + ".npz"
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    if os.path.exists(args.outfile):
        print("results already exist, exiting...")
        sys.exit()

    # get file lengths (number of documents in each file) from doc info
    doc_info = np.load(args.doc_info_file)
    file_lengths = np.bincount(doc_info[:, 0])

    # build and save sparse term-document occurrence matrix
    term_doc_matrix = build_term_document_matrix(args.matches_dir,
                                                 file_lengths,
                                                 args.offset)
    sp.sparse.save_npz(args.outfile, term_doc_matrix)


if __name__ == "__main__":
    main(parser.parse_args())

