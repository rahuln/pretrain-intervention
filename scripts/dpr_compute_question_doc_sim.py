""" given the path to a file with DPR embeddings of a set of instances and
    another path to a directory with a set of .npy files containing embeddings
    of documents, calculate the similarity scores between each instance-
    document pair and save them to a specified output directory """

from argparse import ArgumentParser
from glob import glob
import os
import sys

import numpy as np
from tqdm import tqdm


parser = ArgumentParser(description="encode a question from a dataset with a "
                                    "DPRQuestionEncoder model, then calculate "
                                    "its similarity to a set of encoded "
                                    "documents")
parser.add_argument("--inst_embeddings_file", type=str, required=True,
                    help="path to embeddings of dataset instances")
parser.add_argument("--doc_embeddings_dir", type=str, required=True,
                    help="path to document embeddings files")
parser.add_argument("--outdir", type=str, required=True,
                    help="path to output directory to save similarity scores")


def main(args):
    """ main function """

    # make output directory
    os.makedirs(args.outdir, exist_ok=True)

    # load instance embeddings
    inst_emb = np.load(args.inst_embeddings_file)

    # get list of document embeddings files
    key = lambda x: int(os.path.splitext(os.path.basename(x))[0])
    files = sorted(glob(os.path.join(args.doc_embeddings_dir,
                                     "*.npy")),
                   key=key)

    # iterate through document embeddings files, check if results already
    # exist, if not then compute and save instance-document similarity scores
    for fname in tqdm(files, ascii=True, desc="computing similarities"):
        savename = os.path.join(args.outdir, os.path.basename(fname))
        if os.path.exists(savename):
            continue
        doc_emb = np.load(fname)
        similarities = np.dot(inst_emb, doc_emb.T)
        np.save(savename, similarities)


if __name__ == "__main__":
    main(parser.parse_args())

