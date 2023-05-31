from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
import pytest

from enrichment_models.tasks.normalizer.prompter import NormalizerPrompter


@pytest.fixture
def df_examples() -> pd.DataFrame:
    df = pd.read_csv(
        str(
            Path(__file__).parent.parent.parent
            / "datasets/100_labeled_consumer_transactions.csv"
        )
    )
    return df


@pytest.fixture
def transaction() -> dict[str, Any]:
    return {
        "amount": 5.55,
        "iso_curency_code": "USD",
        "entry_type": "debit",
        "description": "LYFT *TEMP AUTH HOLD",
        "labels": "Ridesharing and taxis",
    }


def test_prompt(transaction: dict[str, Any]):
    prompter = NormalizerPrompter()
    instruction, input = prompter.prompt(transaction)
    assert (
        instruction
        == """Extract the Merchant and Website from the following bank transactions. 
Note: Use ' | ' as a separator between merchant and website
Examples:
CONSUMER TRANSACTION: debit, AMOUNT: 122 USD, DESCRIPTION: BEVERAGES & MOR GILROY CA 11/22
Merchant | Website: BevMo | bevmo.com
CONSUMER TRANSACTION: credit, AMOUNT: 2.99 USD, DESCRIPTION: McDo CAEN St Pierre Refund
Merchant | Website: Mcdonalds | mcdonalds.com"""
    )
    assert (
        input
        == """CONSUMER TRANSACTION: debit, AMOUNT: 5.55 USD, DESCRIPTION: LYFT *TEMP AUTH HOLD
Merchant | Website: """
    )
