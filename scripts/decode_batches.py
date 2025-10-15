""" script to get batches of tokens from Dolma and save them as either decoded
    documents or token IDs """

import argparse
import json
import numpy as np
import os

from cached_path import cached_path
from datetime import datetime
from huggingface_hub import list_repo_refs
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def get_batch_instances(batch_idx: int, batch_size: int,
    global_indices: np.memmap, dataset) -> list[list[int]]:
    """ get a list of lists of token IDs for the given batch """

    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []

    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)

    return batch_instances


def save_decoded_texts(start_batch_idx, end_batch_idx, tokenizer,
    special_token, output_dir, batch_size, global_indices, dataset):
    """ decode tokens from `dataset` from `start_batch_idx` to `end_batch_idx`
        using `tokenizer`, saving the results in `output_dir` """

    global_index = 0
    for batch_idx in tqdm(range(start_batch_idx, end_batch_idx),
                          desc="decoding batches"):

        file_path = os.path.join(output_dir, f"{batch_idx}.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path + ".gz"):
            continue

        # get a batch of token IDs and decode them into text
        batch_instances = get_batch_instances(batch_idx, batch_size,
                                              global_indices, dataset)
        decoded_texts = tokenizer.batch_decode(batch_instances,
                                               skip_special_tokens=False)

        # split text by special token into documents
        text = special_token.join(decoded_texts).replace(
            special_token + special_token, special_token)
        documents = text.split(special_token)

        # write documents to output file
        with open(file_path, "w") as file:
            for doc in documents:
                file.write(json.dumps({"text" : doc}) + "\n")

        # compress output file using gzip
        os.system(f"gzip {file_path}")


def save_token_ids(start_batch_idx, end_batch_idx, output_dir, batch_size,
    global_indices, dataset):
    """ get tokens in `dataset` from `start_batch_idx` to `end_batch_idx`,
        saving the results in `output_dir` """

    global_index = 0
    for batch_idx in tqdm(range(start_batch_idx, end_batch_idx),
                          desc="saving token IDs"):

        file_path = os.path.join(output_dir, f"{batch_idx}.npy")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            continue

        # get a batch of token IDs and save it as an `.npy` file
        batch_instances = get_batch_instances(batch_idx, batch_size,
                                              global_indices, dataset)
        np.save(file_path, np.array(batch_instances, dtype=np.uint32))


def main():
    parser = argparse.ArgumentParser(description="decode and save batches of "
                                                 "data from Dolma")
    parser.add_argument("--data_order_file_path", type=str, required=True,
                        help="path to the data order file")
    parser.add_argument("--train_config_path", type=str, required=True,
                        help="path to the train config file")
    parser.add_argument("--start_batch_idx", type=int, required=True,
                        help="starting batch index")
    parser.add_argument("--end_batch_idx", type=int, required=True,
                        help="ending batch index")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory to save the text files")
    parser.add_argument("--tokenizer_path", type=str,
                        default="allenai/OLMo-2-0425-1B",
                        help="name of or path to the Huggingface tokenizer")
    parser.add_argument("--special_token", type=str,
                        default="<|endoftext|>",
                        help="special token to split the text")
    parser.add_argument("--keep_tokenized", action="store_true",
                        help="save results as token IDs rather than raw text")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("loading config")
    cfg = TrainConfig.load(args.train_config_path)
    print("building dataset")
    dataset = build_memmap_dataset(cfg, cfg.data)
    batch_size = cfg.global_train_batch_size
    print("loading data order indices")
    global_indices = np.memmap(args.data_order_file_path, mode="r+",
                               dtype=np.uint32)

    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                              trust_remote_code=True)

    start = datetime.now()

    if args.keep_tokenized:
        save_token_ids(args.start_batch_idx, args.end_batch_idx,
                       args.output_dir, batch_size, global_indices, dataset)
    else:
        save_decoded_texts(args.start_batch_idx, args.end_batch_idx, tokenizer,
                           args.special_token, args.output_dir, batch_size,
                           global_indices, dataset)

    print("total time:", datetime.now() - start)


if __name__ == "__main__":
    main()

