""" encode a set of documents stored in .json.gz files using a
    DPRContextEncoder model, saving the resulting embeddings """

from argparse import ArgumentParser
import gzip
import json
import os
import sys

import numpy as np
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from tqdm import tqdm


parser = ArgumentParser(description="encode documents with a "
                                    "DPRContextEncoder model")
parser.add_argument("--model_name", type=str, required=True,
                    help="HuggingFace model name for DPRContextEncoder")
parser.add_argument("--input_file", type=str, required=True,
                    help="path to the input .json.gz file")
parser.add_argument("--output_file", type=str, required=True,
                    help="path to save the encoded output")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size for encoding")
parser.add_argument("--max_length", type=int, default=512,
                    help="max length of encoded text")


def main(args):
    """ main function """

    # make output directory, check for result
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    if os.path.exists(args.output_file):
        print("result already exists, exiting...")
        sys.exit()

    # load the tokenizer and model
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(args.model_name)
    model = DPRContextEncoder.from_pretrained(args.model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    encoded_docs = []

    # read documents from file
    with gzip.open(args.input_file, "rt") as f:
        lines = f.readlines()
    docs = [json.loads(line)["text"] for line in lines]

    # encode documents in batches
    for i in tqdm(range(0, len(docs), args.batch_size),
                  ascii=True, ncols=80, desc="encoding documents"):
        batch = docs[i:i + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs).pooler_output
            encoded_docs.append(outputs.cpu().numpy())

    # save the embeddings to output file
    np.save(args.output_file, np.vstack(encoded_docs))


if __name__ == "__main__":
    main(parser.parse_args())

