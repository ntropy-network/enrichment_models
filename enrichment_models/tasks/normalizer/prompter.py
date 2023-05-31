import warnings

import pandas as pd  # type: ignore


class NormalizerPrompter:
    QUESTION_PROMPT: str = "Merchant | Website: "
    SEPARATOR: str = " | "
    NONE_VALUE = "None"

    def _instruction_prompt(self) -> str:
        return """Extract the Merchant and Website from the following bank transactions. 
Note: Use ' | ' as a separator between merchant and website
Examples:
CONSUMER TRANSACTION: debit, AMOUNT: 122 USD, DESCRIPTION: BEVERAGES & MOR GILROY CA 11/22
Merchant | Website: BevMo | bevmo.com
CONSUMER TRANSACTION: credit, AMOUNT: 2.99 USD, DESCRIPTION: McDo CAEN St Pierre Refund
Merchant | Website: Mcdonalds | mcdonalds.com"""

    def _transaction_prompt(self, transaction: dict) -> str:
        iso_currency_code = transaction.get("iso_currency_code", "USD")
        return f"CONSUMER TRANSACTION: {transaction['entry_type']}, AMOUNT: {transaction['amount']} {iso_currency_code}, DESCRIPTION: {transaction['description']}"

    def prompt(
        self,
        transaction: dict,
    ) -> tuple[str, str]:
        instruction = self._instruction_prompt()
        input_transaction = self._transaction_prompt(transaction)
        input = f"""{input_transaction}
{self.QUESTION_PROMPT}"""
        return instruction, input

    def extract_response(self, pred: str) -> dict[str, str]:
        pred = pred.replace(self.QUESTION_PROMPT, "").strip()
        pred = pred.replace("Merchant:", "").strip()  # Fix zero shot gpt4
        pred = pred.replace("Website:", "").strip()  # Fix zero shot gpt4
        merchant_website: list[str] = pred.split(self.SEPARATOR)
        merchant_website = [mw.strip() for mw in merchant_website]
        if len(merchant_website) < 2:  # Template not respected
            warnings.warn(
                f"LLM answer: '{pred}', not respecting template. Replacing by '{self.NONE_VALUE}'."
            )
            return {"merchant": self.NONE_VALUE, "website": self.NONE_VALUE}
        return {"merchant": merchant_website[0], "website": merchant_website[1]}
