"""
Measuring and Improving Consistency in Pretrained Language Models
https://aclanthology.org/2021.tacl-1.60/

Version of the ParaRel dataset formatted for left-to-right language modeling,
completing the object entity in a subject-relation-object phrase.

Homepage: https://huggingface.co/datasets/coastalcph/pararel_patterns
"""
from typing import Optional

from datasets import load_dataset, load_from_disk

from catwalk.dependencies.lm_eval.base import MultipleChoiceTask
from catwalk.task import rc_metrics
from catwalk.tasks.eleuther import EleutherMMLUTask

from .std_fewshot import STD_FEWSHOT
from .utils import make_cloze_prompt, make_mcq_prompt
from .pararel_patterns_info import TEMPLATES

_CITATION = """
@inproceedings{ParaRel2021,
    title={Measuring and Improving Consistency in Pretrained Language Models},
    author={Yanai Elazar and Nora Kassner and Shauli Ravfogel and Abhilasha Ravichander and Eduard Hovy and Hinrich Schütze and Yoav Goldberg},
    booktitle={TACL},
    year={2021}
}
"""


def create_catwalk_pararel_patterns_std_tasks():
    """Creates a dictionary of tasks from a list of relations"""
    res = {}
    for rel in sorted(list(TEMPLATES.keys())):
        for i, template in enumerate(TEMPLATES[rel]):
            rel_name = rel.replace(".jsonl", "").lower()
            res[f"pararel_patterns_std_{rel_name}_temp{i}"] = EleutherMMLUTask(
                f"pararel_patterns_std_{rel_name}_temp{i}",
                ranked_classification=True
            ).add_metrics(rc_metrics(primary="acc_per_token"))
    return res


def create_all_tasks():
    """Creates a dictionary of tasks from a list of relations"""
    res = {}
    for rel in sorted(list(TEMPLATES.keys())):
        for i, template in enumerate(TEMPLATES[rel]):
            rel_name = rel.replace(".jsonl", "").lower()
            res[f"pararel_patterns_std_{rel_name}_temp{i}"] = \
                create_task(rel, template)
    return res


def create_task(relation, template):
    class ParaRelPatternsRelationTest(GeneralParaRelPatternsRelationTest):
        def __init__(self):
            super().__init__(relation, template)

    return ParaRelPatternsRelationTest


class GeneralParaRelPatternsRelationTest(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "coastalcph/pararel_patterns"

    def __init__(self, relation, template):
        self.RELATION = relation
        self.TEMPLATE = template
        super().__init__()

    def download(self):
        self._dataset = load_dataset(self.DATASET_PATH)
        def fn(x):
            return (x["relation"] == self.RELATION) \
                   and (x["template"] == self.TEMPLATE)
        self._dataset["train"] = self._dataset["train"].filter(fn)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def unconditioned_prompt(self):
        return None  # To save compute as the standard will not use unconditional scoring

    def _process_doc(self, doc):
        # Question: Lavoisier Island is located in
        # Answer: Antarctica
        out_doc = {
            "query": make_cloze_prompt(doc["query"]),
            "choices": doc["candidates"],
            "gold": doc["candidates"].index(doc["object"].strip()),
        }
        return out_doc

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't
        if self._fewshot_docs is None:
            self._fewshot_docs: Optional[list] = list(map(self._process_doc, self.dataset["dev"]))

        # use the unchanged order of the dev set without sampling,
        # just as in the original code https://github.com/hendrycks/test/blob/master/evaluate.py#L28
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class ParaRelPatternsSubsetQAStd(MultipleChoiceTask):
    VERSION = 0
    # DATASET_PATH = "coastalcph/pararel_patterns"
    DATASET_PATH = "data/huggingface/pararel_patterns_subset.hf"
    DATASET_NAME = "default"

    def download(self):
        self._dataset = load_from_disk(self.DATASET_PATH)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs: Optional[list] = list(
                map(self._process_doc, self.dataset["train"])
            )
        return self._training_docs

    def _process_doc(self, doc):
        # Question: Lavoisier Island is located in
        # Answer: Antarctica
        out_doc = {
            "id": doc["id"],
            "query": make_cloze_prompt(doc["query"]),
            "choices": doc["candidates"],
            "gold": doc["candidates"].index(doc["object"].strip()),
        }
        return out_doc

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs: Optional[list] = list(
                map(self._process_doc, STD_FEWSHOT[self.DATASET_PATH])
            )
        assert k <= len(self._fewshot_docs)
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class ParaRelPatternsQAStd(ParaRelPatternsSubsetQAStd):
    VERSION = 0
    DATASET_PATH = "coastalcph/pararel_patterns"
    DATASET_NAME = "default"

    def download(self):
        self._dataset = load_dataset(self.DATASET_PATH)

    def _process_doc(self, doc):
        # Question: Lavoisier Island is located in
        # Answer: Antarctica
        out_doc = {
            "query": make_cloze_prompt(doc["query"]),
            "choices": doc["candidates"],
            "gold": doc["candidates"].index(doc["object"].strip()),
        }
        return out_doc

