""" given a dataset, encode a set of instances using a DPRQuestionEncoder
    and save the resulting embeddings """

from argparse import ArgumentParser
import os
import sys

from datasets import load_dataset, load_from_disk, concatenate_datasets
import numpy as np
import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from tqdm import tqdm


parser = ArgumentParser(description="encode instances with a "
                                    "DPRQuestionEncoder model")
parser.add_argument("--model_name", type=str, required=True,
                    help="HuggingFace model name for DPRContextEncoder")
parser.add_argument("--dataset", type=str, required=True,
                    help="name of HuggingFace dataset of questions to load")
parser.add_argument("--data_config", type=str, default=None,
                    help="name of HuggingFace dataset config")
parser.add_argument("--data_subset", type=str, nargs="+",
                    default=["validation"],
                    help="data subset(s) to use")
parser.add_argument("--load_from_disk", action="store_true",
                    help="load dataset from disk rather than HuggingFace "
                         "datasets library")
parser.add_argument("--column_name", type=str, default="question",
                    help="name of column in dataset containing question")
parser.add_argument("--add_answer", action="store_true",
                    help="concatenate answer to question before encoding")
parser.add_argument("--answer_column_name", type=str, default=None,
                    help="name of column in dataset containing correct answer")
parser.add_argument("--indices_file", type=str, default=None,
                    help="path to file with indices of instances to select")
parser.add_argument("--indices_key", type=str, default=None,
                    help="key in indices file for instance indices")
parser.add_argument("--output_file", type=str, required=True,
                    help="path to save the encoded output")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size for encoding")
parser.add_argument("--max_length", type=int, default=512,
                    help="max length of encoded text")


def main(args):
    """ main function """

    # check if embeddings have already been computed 
    if os.path.exists(args.output_file):
        print("result already exists, exiting...")
        sys.exit()
    outdir = os.path.dirname(args.output_file)
    os.makedirs(outdir, exist_ok=True)

    # load the tokenizer and model
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(args.model_name)
    model = DPRQuestionEncoder.from_pretrained(args.model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load dataset
    print("loading dataset...")
    if args.load_from_disk:
        dataset = load_from_disk(args.dataset)
    else:
        args.data_config = None if args.data_config == "None" else args.data_config
        dataset = load_dataset(args.dataset, args.data_config)

    # get subset of dataset
    dataset = concatenate_datasets([dataset[subset] for subset in args.data_subset])

    # get subset of indices if specified
    if args.indices_file is not None and args.indices_key is not None:
        indices = np.load(args.indices_file)
        subset = dataset.select(indices[args.indices_key])
    else:
        subset = dataset

    # get list of questions
    questions = subset[args.column_name]

    # concatenate each question with its answer
    if args.add_answer:
        if args.answer_column_name is None:
            raise ValueError("must specify column name if adding answer")
        # MMLU and OpenBookQA need to be handled differently, since they don't
        # have a single column that contains the correct answer
        if "mmlu" in args.dataset:
            questions = [f"{question} {elem['choices'][elem['answer']]}"
                         for question, elem in zip(questions, subset)]
        elif "openbookqa" in args.dataset:
            answers = [elem["choices"]["text"][elem["choices"]["label"].index(elem["answerKey"])]
                       for elem in subset]
            questions = [f"{question} {answer}" for question, answer
                         in zip(questions, answers)]
        else:
            questions = [f"{question} {answer}" for question, answer
                         in zip(questions, subset[args.answer_column_name])]

    # encode questions in batches
    encoded_questions = []
    for i in tqdm(range(0, len(questions), args.batch_size),
                  ascii=True, ncols=80, desc="encoding questions"):
        batch = questions[i:i + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=args.max_length, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs).pooler_output
            encoded_questions.append(outputs.cpu().numpy())

    # save the embeddings to output file
    np.save(args.output_file, np.vstack(encoded_questions))


if __name__ == "__main__":
    main(parser.parse_args())

