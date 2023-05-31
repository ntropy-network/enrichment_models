import warnings

import pandas as pd  # type: ignore

from enrichment_models.tasks.labeler.hierarchy import (
    CONSUMER_LABELS_CREDIT,
    CONSUMER_LABELS_DEBIT,
    NOT_ENOUGH_INFO_LABEL,
)


class LabelerPrompter:
    QUESTION_PROMPT: str = "Category: "
    ALL_LABELS: list[str] = CONSUMER_LABELS_DEBIT + CONSUMER_LABELS_CREDIT

    def _instruction_prompt(self, entry_type: str) -> str:
        if entry_type in ["credit", "incoming"]:
            candidate_labels = CONSUMER_LABELS_CREDIT
        else:
            candidate_labels = CONSUMER_LABELS_DEBIT
        return f"Classify a bank transaction into one of the following categories (separated by ; ):\n{'; '.join(candidate_labels)}."

    def _transaction_prompt(self, transaction: dict) -> str:
        iso_currency_code = transaction.get("iso_currency_code", "USD")
        return f"CONSUMER TRANSACTION: {transaction['entry_type']}, AMOUNT: {transaction['amount']} {iso_currency_code}, DESCRIPTION: {transaction['description']}"

    def prompt(
        self,
        transaction: dict,
    ) -> tuple[str, str]:
        entry_type: str = transaction["entry_type"]
        instruction = self._instruction_prompt(entry_type=entry_type)
        input_transaction = self._transaction_prompt(transaction)
        input = f"""{input_transaction}
{self.QUESTION_PROMPT}"""
        return instruction, input

    def extract_response(self, pred: str) -> dict:
        pred = pred.replace(self.QUESTION_PROMPT, "").strip()
        if pred[-1] in [".", ";"]:  # Case when LLM add a punctuation point at the end
            pred = pred[:-1]
        # If output doesn't match the list of possible labels,
        # replace with "Not enough information"
        if pred not in self.ALL_LABELS:
            warnings.warn(
                f"LLM answer: '{pred}' not in label list, replacing by '{NOT_ENOUGH_INFO_LABEL}'."
            )
            pred = NOT_ENOUGH_INFO_LABEL
        return {"labels": pred}
