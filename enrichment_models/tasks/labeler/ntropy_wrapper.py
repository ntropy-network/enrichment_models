import os
import uuid
from typing import Any

import pandas as pd  # type: ignore
from ntropy_sdk import SDK  # type: ignore


class NtropyLabeler:
    def __init__(self, api_key: str | None = None) -> None:
        if api_key is None:
            api_key = os.environ["NTROPY_API_KEY"]
        self.sdk = SDK(api_key)

    def predict(self, transactions: pd.DataFrame | list[dict[str, Any]]) -> list[dict]:
        if isinstance(transactions, list):
            transactions = pd.DataFrame.from_records(transactions)

        transactions["account_holder_id"] = str(uuid.uuid4())
        if "iso_currency_code" not in transactions.columns:
            transactions["iso_currency_code"] = "USD"
        transactions["account_holder_id"] = str(uuid.uuid4())
        enriched_df = self.sdk.add_transactions(
            transactions[
                [
                    "account_holder_id",
                    "account_holder_type",
                    "date",
                    "amount",
                    "entry_type",
                    "description",
                    "iso_currency_code",
                ]
            ]
        )

        preds: list[str] | list[list[str]] = enriched_df["labels"].tolist()
        if isinstance(preds[0], list):
            preds = [" - ".join(pred) for pred in preds]
        return [{"labels": pred} for pred in preds]
