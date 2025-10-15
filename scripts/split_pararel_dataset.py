""" script to split ParaRel Patterns dataset into subsets by relation and
    save each subset as a separate HuggingFace dataset """

from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
import json
import os

from datasets import load_dataset, load_from_disk
import numpy as np
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--templates_file", type=str,
                    default="data/pararel_patterns_templates.json",
                    help="path to file with mapping from relation to template")
parser.add_argument("--outdir", type=str,
                    default="data/huggingface",
                    help="path to output directory to save data subsets")
parser.add_argument("--subdir_pattern", type=str,
                    default="pararel_patterns_{relation}",
                    help="pattern of output directory for data subset")


def main(args):
    """ main function """

    # load dataset, get list of relations
    dataset = load_dataset("coastalcph/pararel_patterns")
    relations = sorted(set(dataset["train"]["relation"]))

    if not os.path.exists(args.templates_file):
        print("Creating relation-to-template mapping\n")

        # get all templates for each relation
        relation_to_templates = defaultdict(set)
        for inst in tqdm(dataset["train"]):
            relation_to_templates[inst["relation"]].add(inst["template"])
        relation_to_templates = {k : sorted(v) for k, v
                                 in relation_to_templates.items()}

        # prompt user to select single template for each relation
        for rel, templates in tqdm(relation_to_templates.items()):
            for i, template in enumerate(templates):
                print(f"[{i}] {template}")
            choice = int(input("choice: "))
            relation_to_templates[rel] = templates[choice]
            print("\n")

        # save mapping from relation to single template
        with open(args.templates_file, "w") as f:
            json.dump(relation_to_templates, f, indent=4)

    else:
        # load mapping from relation to template from file
        print("Loading relation-to-template mapping")
        with open(args.templates_file, "r") as f:
            relation_to_templates = json.load(f)

    # filter entire dataset for each {relation, template} pair and save that
    # subset as a separate dataset
    for relation, template in tqdm(relation_to_templates.items()):

        # check for saved subset, create output directory
        rel_str = relation.replace(".jsonl", "").lower()
        subdir = args.subdir_pattern.format(relation=rel_str)
        output_dir = os.path.join(args.outdir, subdir)
        if os.path.exists(output_dir):
            continue
        os.makedirs(args.outdir, exist_ok=True)

        # filtering function 
        def fn(row):
            return ((row["relation"] == relation)
                    and (row["template"] == template))

        # create data subset for relation, save to output directory
        data_subset = deepcopy(dataset)
        data_subset["train"] = data_subset["train"].filter(fn)
        ids = np.arange(len(data_subset["train"]))
        data_subset["train"] = data_subset["train"].add_column("id", ids)
        data_subset.save_to_disk(output_dir)


if __name__ == "__main__":
    main(parser.parse_args())

