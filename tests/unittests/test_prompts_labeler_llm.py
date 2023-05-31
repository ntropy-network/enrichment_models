from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
import pytest

from enrichment_models.tasks.labeler.prompter import LabelerPrompter


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
    prompter = LabelerPrompter()
    instruction, input = prompter.prompt(transaction)
    assert (
        instruction
        == """Classify a bank transaction into one of the following categories (separated by ; ):
ATM/bank withdrawal; App stores; Auto lease payment; Auto loan repayment; Bank fee; Books, newsletters, newspapers; Buy now, pay later; Cafes and coffee shops; Childcare; Clothing; Contribution to reserve fund; Convenience stores; Council tax; Credit card bill; Credit card fee; Credit report; Debt collection; Department and discount stores; Donation; Drugstores and pharmacies; Education; Electronics; Entertainment and recreation; Firearms; Food and Drink; Fuel; Funerals and bequests; Gambling; Gifts; Government; Groceries; Home improvements and maintenance services; Hotels and lodging; Insurance; Inter account transfer; Interest; Intra account transfer; Investment; Laundry; Legal services; Liquor; Loan repayment; Media; Medical bill; Mortgage; Non-sufficient funds / Overdraft fee; Not enough information; Other consumer services; Other non-essential; Other transport; Pawn shops; Peer to peer transfer; Pets; Prenote; Public transport; Recreational goods; Rent and property management fee; Rent to own; Retirement contributions; Reversal / adjustment; Ridesharing and taxis; SaaS tools; Self care; Sport and fitness; Student loan repayment; Taxes; Toll charge; Towing companies; Trading (crypto); Trading (non-crypto); Utilities; Vehicle maintenance; eCommerce purchase."""
    )
    assert (
        input
        == """CONSUMER TRANSACTION: debit, AMOUNT: 5.55 USD, DESCRIPTION: LYFT *TEMP AUTH HOLD
Category: """
    )
