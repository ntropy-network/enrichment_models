from typing import Any

import pandas as pd  # type: ignore

from enrichment_models.llms.llm import LLM
from enrichment_models.tasks.labeler.prompter import LabelerPrompter
from enrichment_models.tasks.normalizer.prompter import NormalizerPrompter


class LLMWrapper:
    def __init__(
        self,
        llm: LLM,
        prompter: LabelerPrompter | NormalizerPrompter,
    ) -> None:
        self.llm = llm
        self.prompter = prompter

    def predict(self, transactions: pd.DataFrame | list[dict[str, Any]]) -> list[dict]:
        if isinstance(transactions, list):
            transactions = pd.DataFrame.from_records(transactions)
        else:
            transactions = transactions.copy()

        # get all instructions/inputs tuples
        transactions[["instruction", "input"]] = transactions.apply(
            self.prompter.prompt, axis=1, result_type="expand"
        )

        # make inference
        preds = self.llm.predict(
            instructions=transactions["instruction"].to_list(),
            inputs=transactions["input"].to_list(),
        )
        assert isinstance(preds, list)

        # extract llm answer/prediction
        return [self.prompter.extract_response(pred) for pred in preds]
