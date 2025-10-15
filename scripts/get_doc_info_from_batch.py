""" given a path to a set of files each containing a batch of token IDs, get
    information on each document in each batch, where the information consists
    of the batch number, sequence number (in the batch), and start and end
    position in the sequence (not including the EOS tokens that separate
    documents) """

from argparse import ArgumentParser
import ctypes
from glob import glob
import os
import sys

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


# command-line arguments
parser = ArgumentParser(description="get info on docs in batches of tokens")
parser.add_argument("--datadir", type=str, required=True,
                    help="path to directory containing token batch files")
parser.add_argument("--tokenizer", type=str, default="allenai/OLMo-2-0425-1B",
                    help="name of Huggingface tokenizer")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to file to save document info")


# Load the shared library
convolution_lib = ctypes.CDLL('./scripts/convolution.so')

# Define the function argument types and return type
convolution_lib.convolve.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
]

def convolve(signal, kernel):
    signal = np.array(signal, dtype=np.float64)
    kernel = np.array(kernel, dtype=np.float64)

    signal_len = signal.shape[0]
    kernel_len = kernel.shape[0]

    output_len = signal_len - kernel_len + 1
    output = np.zeros(output_len, dtype=np.float64)

    # Call the C function
    convolution_lib.convolve(
        signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), signal_len,
        kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), kernel_len,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    return output


def main(args):
    """ main function """

    # check for output file, make output directory
    outdir = os.path.dirname(args.outfile)
    os.makedirs(outdir, exist_ok=True)
    if os.path.exists(args.outfile):
        print("result already exists, exiting...")
        sys.exit()

    # get data batch files
    key = lambda x: int(os.path.splitext(os.path.basename(x))[0])
    files = sorted(glob(os.path.join(args.datadir, "*.npy")), key=key)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              trust_remote_code=True)

    print(vars(args))

    doc_info = list()
    desc = "collecting document info"
    for batch_idx, fname in enumerate(tqdm(files, desc=desc)):
        batch = np.load(fname)

        for seq_idx, seq in enumerate(batch):

            # get indices of EOS token IDs
            eos_token_idxs = list(np.where(seq == tokenizer.eos_token_id)[0])

            # check for consecutive eos_token_id values (should not occur)
            idx_diffs = np.diff(eos_token_idxs)
            for i in range(len(idx_diffs) - 1):
                if idx_diffs[i] == 1 and idx_diffs[i + 1] == 1:
                    raise RuntimeError("consecutive EOS tokens found")

            # EDGE CASE: really strange special case for OLMo 2 tokenizer, but
            # in some batches the "<|endoftext|>" token doesn't appear as a
            # single token, but gets split up into
            #     "<" "|" "endo" "ft" "ext" "|" ">"
            # where the first and last symbols sometimes vary. Since this was
            # treated as a valid EOS token when splitting documents in their
            # raw text form, we can address this by searching for the token IDs
            # of the 3-token sequence in the middle and replacing it (and the
            # token IDs immediately before and after it) with the EOS token ID,
            # which will just be ignored in the for-loop below
            if "OLMo-2" in args.tokenizer:
                kernel = np.array([8862, 728, 428])  # "endo", "ft", "ext"
                matches = convolve(seq, kernel)
                match_idxs = np.where(matches == len(kernel))[0]
                for i in match_idxs:
                    subseq = seq[i - 2 : i + 5]
                    # make sure it's the whole EOS token, not just a portion
                    if tokenizer.eos_token in tokenizer.decode(subseq):
                        loc = (batch_idx, seq_idx, i - 2, i + 5)
                        print("FOUND SPLIT EOS TOKEN", loc)
                        for j in range(i - 2, i + 5):
                            eos_token_idxs.append(j)
                eos_token_idxs = sorted(eos_token_idxs)

            # if there are no EOS token IDs, the entire sequence is a single
            # document
            if len(eos_token_idxs) == 0:
                doc_info.append([batch_idx, seq_idx, 0, len(seq) - 1])
                continue

            # if sequence does not end in an EOS token ID, add length of
            # sequence to end of indices
            if eos_token_idxs[-1] != len(seq) - 1:
                eos_token_idxs = eos_token_idxs + [len(seq)]

            # if sequence does not start with an EOS token ID, prepend -1
            if eos_token_idxs[0] != 0:
                eos_token_idxs = [-1] + eos_token_idxs

            # EDGE CASE: empty doc at beginning of batch
            if seq_idx == 0 and seq[0] == tokenizer.eos_token_id:
                doc_info.append([batch_idx, seq_idx, 0, 0])

            # for each document, get its start and end token index (not
            # including EOS token ID indices), add to document info list
            for idx in range(len(eos_token_idxs) - 1):
                start_idx = eos_token_idxs[idx] + 1
                end_idx = eos_token_idxs[idx + 1] - 1

                # this ensures we skip "empty" documents (i.e., cases where
                # token pair "<|endoftext|><|endoftext|>" appears in the
                # middle of a batch)
                if start_idx > end_idx:
                    continue

                doc_info.append([batch_idx, seq_idx, start_idx, end_idx])

            # EDGE CASE: empty doc at end of batch
            if seq_idx == len(batch) - 1 and seq[-1] == tokenizer.eos_token_id:
                doc_info.append([batch_idx, seq_idx, len(seq), len(seq)])

            # EDGE CASE: empty doc between sequences
            if seq_idx < len(batch) - 1 and seq[-1] == tokenizer.eos_token_id \
                and batch[seq_idx + 1][0] == tokenizer.eos_token_id:
                doc_info.append([batch_idx, seq_idx, len(seq), len(seq)])

    # save document info to file
    np.save(args.outfile, np.array(doc_info, dtype=np.uint16))


if __name__ == "__main__":
    main(parser.parse_args())

