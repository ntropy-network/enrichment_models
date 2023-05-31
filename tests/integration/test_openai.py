import pytest

from enrichment_models.llms.openai import ChatGPT
from enrichment_models.tasks.labeler.utils import label_similarity_score


def test_():
    model = ChatGPT(model_name="gpt-3.5-turbo", temperature=0)
    res = model.predict(
        instructions="I have a question.",
        inputs="In which year the french revolution happened? (note: just return the date as, nothing else)",
    )
    # remove punctuation (ex: a dot at the end)
    res = "".join(c for c in res if c.isdigit())
    assert res == "1789"


def test_chatgpt():
    model = ChatGPT(model_name="gpt-4", temperature=0)
    res = model.predict(
        instructions="I have a question.",
        inputs="In which year the french revolution happened? (note: just return the date, nothing else)",
    )
    # remove punctuation (ex: a dot at the end)
    res = "".join(c for c in res if c.isdigit())
    assert res == "1789"


def test_label_similarity_score():
    preds = [
        "App stores",
        "Buy now, pay later",
        "Food and Drink",
        "Clothing",
        "Cafes and coffee shops",
        "ATM/bank withdrawal",
        "Firearms",
        "Loan repayment",
        "Mortgage",
        "Mortgage",
        "Auto lease payment",
        "Property rental",
        "Education",
        "Gambling",
        "Insurance",
    ]
    ground_truths = [
        "Software",
        "Books, newsletters, newspapers",
        "Liquor",
        "Recreational goods",
        "Food and Drink",
        "Food and Drink",
        "Childcare",
        "Laundry",
        "Mortgage",
        "Insurance",
        "Auto loan repayment",
        "Refunds",
        "Electronics",
        "Media",
        "Sport and fitness",
    ]
    scores = label_similarity_score(preds, ground_truths, average_reduction=False)
    assert scores == pytest.approx(
        [
            0.9,
            0.09,
            0.9,
            0.5,
            0.9,
            0.14,
            0.05,
            0.12,
            1,
            0.9,
            0.9,
            0.04,
            0.34,
            0.40,
            0.21,
        ],
        abs=0.01,
    )
