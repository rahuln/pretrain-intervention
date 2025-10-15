""" Given a path to a JSON file containing questions, their correct answers,
    and potentially relevant documents, send API calls to an OpenAI model to
    determine if each document is relevant to each question-answer pair (i.e.,
    the document contains enough information to correctly answer the question)
    """

from argparse import ArgumentParser
import json
import os
import sys

from openai import OpenAI
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument("--input-file", type=str, required=True,
                    help="path to JSON file with question-answer-doc inputs")
parser.add_argument("--outfile", type=str, required=True,
                    help="path to output file to save responses")
parser.add_argument("--limit", type=int, default=None,
                    help="limit on number of queries (for debugging)")
parser.add_argument("--prompt", type=str, default="correct_answer",
                    choices=["correct_answer", "educated_guess",
                             "related_topic"],
                    help="prompt to use when querying the OpenAI model")
parser.add_argument("--model", type=str, default="gpt-5",
                    help="name of OpenAI API model to use")
parser.add_argument("--reasoning-effort", type=str, default="minimal",
                    help="level of reasoning to use")
parser.add_argument("--verbosity", type=str, default="low",
                    help="level of verbosity (i.e., number of output tokens)")


prompt_templates = {
    "correct_answer" : ("You will be given a question or question stem and its"
                       " correct answer, along with a document. Output \"yes\""
                       " if the document contains enough information to"
                       " correctly answer the question, and \"no\" otherwise."
                       " Only output \"yes\" or \"no\".\n\n"
                       "Question: {question}\n"
                       "Answer: {answer}\n"
                       "\n"
                       "Document: {document}"),
    "educated_guess" : ("You will be given a question or question stem and its"
                        " correct answer, along with a document. Output "
                        "\"yes\" if the document contains enough information "
                        "to make an educated guess about the answer, and "
                        "\"no\" otherwise. The document does not have to "
                        "have enough information to be able to correctly "
                        "answer the question. Only output \"yes\" or \"no\"."
                        "\n\nQuestion: {question}\n"
                        "Answer: {answer}\n"
                        "\n"
                        "Document: {document}"),
    "related_topic" : ("You will be given a question or question stem and its "
                       "correct answer, along with a document. Output "
                       "\"yes\" if the document is topically related to the "
                       "question and answer, and \"no\" otherwise. The "
                       "document does not have to have enough information to "
                       "to be able to correctly answer the question. Only "
                       "output \"yes\" or \"no\"."
                       "\n\nQuestion: {question}\n"
                       "Answer: {answer}\n"
                       "\n"
                       "Document: {document}")
}


def main(args):
    """ main function """

    # check if output file already exists, make directory if not
    if os.path.exists(args.outfile):
        print("result already exists, exiting...")
        sys.exit()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    # initialize OpenAI client
    client = OpenAI()

    # load queries (question-answer-document triples)
    with open(args.input_file, "r") as f:
        results = json.load(f)
    queries = results["results"]
    if args.limit is not None:
        queries = queries[:args.limit]

    # print example prompt with first query
    print("\nExample prompt:")
    prompt_template = prompt_templates[args.prompt]
    print(prompt_template.format(**queries[0]))
    print()

    # submit queries, collect responses
    responses = list()
    print(f"{args.prompt}, {args.model}, {args.reasoning_effort}, "
          f"{args.verbosity}")
    for query in tqdm(queries, ascii=True, ncols=100, desc="running queries"):
        prompt = prompt_template.format(**query)
        response = client.responses.create(
            model=args.model,
            input=prompt,
            reasoning={ "effort": args.reasoning_effort },
            text={ "verbosity": args.verbosity },
        )
        responses.append(json.loads(response.to_json()))

    # construct and save final results
    results = {"responses" : responses, "args" : vars(args)}
    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main(parser.parse_args())
