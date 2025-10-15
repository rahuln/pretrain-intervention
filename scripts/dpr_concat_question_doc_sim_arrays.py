""" given a path to a set of numpy arrays containing question-document DPR
    similarity scores, concatenate them into a single matrix of shape
    (num_questions, num_documents), optionally keeping track of only the top-k
    highest similarity scores per question and saving as a scipy sparse matrix
    """

from argparse import ArgumentParser
from glob import glob
import os
import sys

import numpy as np
import scipy as sp
from tqdm import tqdm


parser = ArgumentParser(description="concatenate DPR score matrices")
parser.add_argument("--path-to-scores", type=str, required=True,
                    help="path to DPR score matrix files")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to output file")
parser.add_argument("--start-index", type=int, default=None,
                    help="index of first file to use (inclusive)")
parser.add_argument("--end-index", type=int, default=None,
                    help="index of last file to use (exclusive)")
parser.add_argument("--k", type=int, default=None,
                    help="number of top scores to keep per question, saving "
                         "the matrix as a scipy.sparse.coo_matrix")


def main(args):
    """ main function """

    # create output directory, check for existing output file
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    if os.path.exists(args.outfile):
        print("result already exists, exiting...")
        sys.exit()

    # get sorted files in directory
    key = lambda x: int(os.path.splitext(os.path.basename(x))[0])
    files = sorted(glob(os.path.join(args.path_to_scores, "*.npy")), key=key)
    start = args.start_index if args.start_index is not None else 0
    end = args.end_index if args.end_index is not None else len(files)
    print(f"using {end - start} / {len(files)} files")
    files = files[start : end]

    # load arrays from files
    arrays = list()
    for fname in tqdm(files, ascii=True, ncols=100, desc="loading files"):
        arrays.append(np.load(fname))
    matrix = np.hstack(arrays)

    # save as numpy array or coo_matrix, depending on if args.k is specified
    if args.k is not None:

        # get data, row and col indices for sparse matrix
        data, row, col = list(), list(), list()
        for r, scores in enumerate(tqdm(matrix, ascii=True, ncols=100,
                                        desc="creating sparse matrix")):
            top_col = np.argsort(scores)[::-1][:args.k]
            data.extend(scores[top_col])
            row.extend([r] * args.k)
            col.extend(top_col)
        matrix = sp.sparse.coo_matrix((data, (row, col)), shape=matrix.shape)

        if not args.outfile.endswith(".npz"):
            args.outfile = args.outfile + ".npz"
        sp.sparse.save_npz(args.outfile, matrix)
    else:
        if not args.outfile.endswith(".npy"):
            args.outfile = args.outfile + ".npy"
        np.save(args.outfile, matrix)


if __name__ == "__main__":
    main(parser.parse_args())

