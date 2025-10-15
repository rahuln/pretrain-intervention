""" swap out sequences of tokens corresponding to documents in a batch of data
    with specified token sequences in another batch, creating and saving a new
    batch of data """


from argparse import ArgumentParser
import os
import pickle
from pprint import pprint
import random

import numpy as np
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser(description="create a new batch of data by swapping "
                                    "documents with another batch")
parser.add_argument("--doc_idxs_file", type=str, required=True,
                    help="path to file containing indices of original, new, "
                         "and extra documents for swapping")
parser.add_argument("--doc_idxs_mapping", type=str, default=None,
                    help="saved dict mapping between orig and new doc idxs")
parser.add_argument("--path_to_orig_data_dir", type=str, required=True,
                    help="path to directory containing original data batch")
parser.add_argument("--path_to_new_data_dir", type=str, required=True,
                    help="path to directory containing new data batch")
parser.add_argument("--path_to_orig_doc_info", type=str, required=True,
                    help="path to document info file for original batch")
parser.add_argument("--path_to_new_doc_info", type=str, required=True,
                    help="path to document info file for new batch")
parser.add_argument("--orig_offset", type=int, required=True,
                    help="offset for filename of original files")
parser.add_argument("--new_offset", type=int, required=True,
                    help="offset for filename of new files")
parser.add_argument("--outdir", type=str, required=True,
                    help="path to output directory")
parser.add_argument("--skip_extra_docs", action="store_true",
                    help="don't use extra docs to replace orig docs")
parser.add_argument("--pair_by_new_first", action="store_true",
                    help="prioritize matching sizes of new docs")
parser.add_argument("--seed", type=int, default=None,
                    help="random seed")


def map_doc_idxs(args):
    """ helper function to find mapping between original and new / extra doc,
        indices matching token counts as much as possible """

    # load doc indices files, get indices of original, new, and extra documents
    # (extra documents are used to replace original documents if there aren't
    # enough new documents to do so)
    doc_idxs = np.load(args.doc_idxs_file)
    orig_idxs = np.array(sorted(doc_idxs["orig_idxs"]))
    new_idxs = np.array(sorted(doc_idxs["new_idxs"]))
    extra_idxs = np.array(sorted(doc_idxs["extra_idxs"]))

    # load document info files (contain batch index, sequence index, start
    # token, and stop token - inclusive - for every document in the batch)
    orig_info = np.load(args.path_to_orig_doc_info)
    new_info = np.load(args.path_to_new_doc_info)

    # get document sizes for all original, new, and extra documents
    orig_idxs_and_sizes = [(idx, orig_info[idx, -1] - orig_info[idx, -2] + 1)
                           for idx in orig_idxs]
    new_idxs_and_sizes = [(idx, new_info[idx, -1] - new_info[idx, -2] + 1)
                          for idx in new_idxs]
    extra_idxs_and_sizes = [(idx, new_info[idx, -1] - new_info[idx, -2] + 1)
                            for idx in extra_idxs]

    # shuffle each of the arrays so that when they're sorted by document
    # length, tie-breaking lengths doesn't favor any ordering of batches
    # (otherwise, in the case where there are more orig docs than new docs,
    # there's a risk that most of the replaced docs are in a small subset of
    # the batches)
    if args.seed is not None:
        random.shuffle(orig_idxs_and_sizes)
        random.shuffle(new_idxs_and_sizes)
        random.shuffle(extra_idxs_and_sizes)

    orig_idxs_and_sizes = sorted(orig_idxs_and_sizes, key=lambda x: x[1])
    new_idxs_and_sizes = sorted(new_idxs_and_sizes, key=lambda x: x[1])
    extra_idxs_and_sizes = sorted(extra_idxs_and_sizes, key=lambda x: x[1])

    # match original documents to new documents of equal or greater size
    orig_to_new_idx = dict()
    pos = 0
    offsets = list()
    no_more_docs = False
    last_orig_idx = 0
    desc = "matching original with new documents"

    if args.pair_by_new_first:
        # prioritize matching lengths of new docs, which requires iterating
        # through documents from largest-to-smallest (i.e., for each new doc,
        # we want to find the largest orig doc that is equal in size or
        # smaller than it)
        print("pairing by new docs first")
        orig_idxs_and_sizes = orig_idxs_and_sizes[::-1]
        new_idxs_and_sizes = new_idxs_and_sizes[::-1]
        if len(orig_idxs_and_sizes) > 0:
            for idx, (new_idx, new_size) in enumerate(tqdm(new_idxs_and_sizes,
                                                           desc=desc)):
                while new_size < orig_idxs_and_sizes[pos][1]:
                    pos += 1
                    if pos >= len(orig_idxs_and_sizes):
                        last_orig_idx = pos
                        no_more_docs = True
                        break
                if no_more_docs:
                    break
                orig_to_new_idx[orig_idxs_and_sizes[pos][0]] = new_idx
                offsets.append(orig_idxs_and_sizes[pos][1] - new_size)
                pos += 1
                if pos >= len(orig_idxs_and_sizes):
                    last_orig_idx = pos
                    no_more_docs = True
                    break
        # reverse lists to be least-to-greatest size again, fix index
        orig_idxs_and_sizes = orig_idxs_and_sizes[::-1]
        new_idxs_and_sizes = new_idxs_and_sizes[::-1]
        last_orig_idx = len(orig_idxs_and_sizes) - last_orig_idx
    else:
        if len(new_idxs_and_sizes) > 0:
            for idx, (orig_idx, orig_size) in enumerate(tqdm(orig_idxs_and_sizes,
                                                             desc=desc)):
                while new_idxs_and_sizes[pos][1] < orig_size:
                    pos += 1
                    if pos >= len(new_idxs_and_sizes):
                        last_orig_idx = idx
                        no_more_docs = True
                        break
                if no_more_docs:
                    break
                orig_to_new_idx[orig_idx] = new_idxs_and_sizes[pos][0]
                offsets.append(new_idxs_and_sizes[pos][1] - orig_size)
                pos += 1
                if pos >= len(new_idxs_and_sizes):
                    last_orig_idx = idx
                    no_more_docs = True
                    break

    # match remaining original documents to extra documents of equal or greater
    # size (this loop is written in a way that assumes we have enough extra
    # documents and won't run out)
    if not args.skip_extra_docs:
        pos = 0
        desc = "matching original with extra documents"
        for idx in tqdm(range(last_orig_idx, len(orig_idxs_and_sizes)),
                        desc=desc):
            orig_idx, orig_size = orig_idxs_and_sizes[idx]
            while extra_idxs_and_sizes[pos][1] < orig_size:
                pos += 1
            orig_to_new_idx[orig_idx] = extra_idxs_and_sizes[pos][0]
            offsets.append(extra_idxs_and_sizes[pos][1] - orig_size)
            pos += 1
    else:
        print("skipping extra docs")

    # print stats on offsets
    print(f"offsets mean ± std: {np.mean(offsets):.3f} ± {np.std(offsets):.3f}")

    # print distribution of offsets (i.e., difference in number of tokens
    # between original document and new document that's replacing it)
    # offset_distn = {k : 100 * v / len(offsets) for k, v
    #                 in enumerate(np.bincount(offsets))}
    # pprint(offset_distn)
    basename, _ = os.path.splitext(os.path.basename(args.doc_idxs_file))
    np.save(f"offsets_{basename}.npy", np.array(offsets))
    np.savez(f"orig_to_new_idx_{basename}.npz",
             orig_idxs=np.array(list(orig_to_new_idx.keys())),
             new_idxs=np.array(list(orig_to_new_idx.values())))

    return orig_to_new_idx, orig_idxs, orig_info, new_info, offsets


def main(args):
    """ main function """

    if args.seed is not None:
        random.seed(args.seed)

    # if path to mapping specified, load it, otherwise create mapping
    if args.doc_idxs_mapping is not None:
        print("loading saved doc idx mapping...")

        # load mapping from either .npz or .pkl file
        if args.doc_idxs_mapping.endswith(".npz"):
            idxs = np.load(args.doc_idxs_mapping)
            orig_to_new_idx = {orig_idx : new_idx for orig_idx, new_idx
                               in zip(idxs["orig_idxs"], idxs["new_idxs"])}
            orig_idxs = np.array(sorted(idxs["orig_idxs"]))
        elif args.doc_idxs_mapping.endswith(".pkl"):
            with open(args.doc_idxs_mapping, "rb") as f:
                orig_to_new_idx = pickle.load(f)
            orig_idxs = np.array(sorted(orig_to_new_idx.keys()))

        # load document info files (contain batch index, sequence index, start
        # token, and stop token - inclusive - for every document in the batch)
        orig_info = np.load(args.path_to_orig_doc_info).astype(np.int64)
        orig_num_tokens = orig_info[:, -1] - orig_info[:, -2] + 1
        new_info = np.load(args.path_to_new_doc_info)
        new_num_tokens = new_info[:, -1] - new_info[:, -2] + 1

        # construct offsets from mapping
        offsets = np.array([new_num_tokens[new_idx] - orig_num_tokens[orig_idx]
                            for orig_idx, new_idx in orig_to_new_idx.items()])

    else:
        print("building doc idx mapping...")
        orig_to_new_idx, orig_idxs, orig_info, new_info, offsets = \
            map_doc_idxs(args)

    # print number of total tokens that need to be loaded
    ntok_orig = np.sum(orig_info[orig_idxs, -1] - orig_info[orig_idxs, -2] + 1)
    ntok_new = np.sum(offsets)
    print("num tokens to load:", ntok_orig + ntok_new)

    # load token sequences for all new and extra documents being used
    curr_batch = 0
    fname = f"{args.new_offset}.npy"
    batch = np.load(os.path.join(args.path_to_new_data_dir, fname))
    new_and_extra_idxs = sorted(list(orig_to_new_idx.values()))
    new_tokens = dict()
    desc = "loading tokens for new documents"
    for idx in tqdm(new_and_extra_idxs, desc=desc):
        batch_idx, seq_idx, start, end = new_info[idx]
        if batch_idx != curr_batch:
            del batch
            fname = f"{batch_idx + args.new_offset}.npy"
            batch = np.load(os.path.join(args.path_to_new_data_dir, fname))
        # have to make a copy, otherwise entire array is kept in memory
        new_tokens[idx] = batch[seq_idx, start : end + 1].copy()

    # iterating through files in original data batch, replace token sequences
    # for original documents with new or extra documents as specified by
    # constructed pairings, saving each modified file
    os.makedirs(args.outdir, exist_ok=True)
    curr_open_batch = 0
    curr_batch, curr_seq = 0, 0
    seq_offset = 0
    curr_fname = f"{args.orig_offset}.npy"
    batch = np.load(os.path.join(args.path_to_orig_data_dir, curr_fname))
    desc = "replacing documents"
    num_skipped_docs = 0
    for idx in tqdm(orig_idxs, desc=desc):

        # if we're not replacing this document, skip it
        if idx not in orig_to_new_idx:
            continue

        batch_idx, seq_idx, start, end = orig_info[idx]

        # if in a new batch or sequence, reset the sequence shift offset
        if (batch_idx, seq_idx) != (curr_batch, curr_seq):
            curr_batch, curr_seq = batch_idx, seq_idx
            seq_offset = 0

        # if in a new batch, save the modified batch and load the new one
        if curr_open_batch != curr_batch:
            np.save(os.path.join(args.outdir, curr_fname), batch)
            curr_fname = f"{curr_batch + args.orig_offset}.npy"
            batch = np.load(os.path.join(args.path_to_orig_data_dir,
                                         curr_fname))
            curr_open_batch = curr_batch

        # modify original sequence of tokens by replacing the subsequence
        # corresponding to the original document with the tokens for the new
        # document, shifting all other tokens to the right and truncating to
        # the original sequence length
        new_seq = np.concatenate((batch[seq_idx, :start + seq_offset],
                                  new_tokens[orig_to_new_idx[idx]],
                                  batch[seq_idx, end + seq_offset + 1:]))

        # this try-except is here in case our mapping somehow maps an original
        # document to a new one with fewer tokens, in which case the assignment
        # of the modified new sequence to the row of the batch matrix will fail
        # (shorter new documents will still work, due to the truncation)
        try:
            batch[seq_idx] = new_seq[:len(batch[seq_idx])]

            # keep track of how much the remaining tokens were shifted by, in case
            # we're also replacing a document later in the same sequence
            seq_offset += len(new_tokens[orig_to_new_idx[idx]]) - (end - start + 1)
        except ValueError:
            num_skipped_docs += 1

    # save final batch
    np.save(os.path.join(args.outdir, curr_fname), batch)

    print(f"skipped {num_skipped_docs} documents(s)")
    print("done")


if __name__ == "__main__":
    main(parser.parse_args())
