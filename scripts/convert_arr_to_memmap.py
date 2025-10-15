""" convert all numpy arrays in a directory to memory-mapped arrays """

from argparse import ArgumentParser
from glob import glob
import os
import sys

import numpy as np
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("basedir", type=str,
                    help="path to directory containing .npy files")
parser.add_argument("--index", type=int, default=0,
                    help="which batch of files to process")
parser.add_argument("--batch_size", type=int, default=250,
                    help="batch size for processing files in batches")
parser.add_argument("--suffix", type=str, default="_mmap",
                    help="suffix of new directory with memory-mapped files")


def main(args):
    """ main function """

    # get all .npy files in directory, recursively
    files = sorted(glob(os.path.join(args.basedir, "**", "*.npy"),
                        recursive=True),
                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # get file paths for this batch
    start = args.index * args.batch_size
    end = (args.index + 1) * args.batch_size
    files = files[start : end]

    # iterate through files and convert to memory-mapped array
    dirname = os.path.basename(args.basedir)
    for fname in tqdm(files, ncols=100, ascii=True):
        newname = fname.replace(dirname, f"{dirname}{args.suffix}")
        os.makedirs(os.path.dirname(newname), exist_ok=True)
        arr = np.load(fname)
        mmap_arr = np.memmap(newname, dtype=np.uint32, mode="w+",
                             shape=arr.shape)
        mmap_arr[:] = arr[:]
        mmap_arr.flush()
        del mmap_arr


if __name__ == "__main__":
    main(parser.parse_args())
