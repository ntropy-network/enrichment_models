from urllib.parse import urlsplit

import pandas as pd  # type: ignore


def clean_url(url: str) -> str:
    parsed_url = urlsplit(url)

    # If string is an url
    if parsed_url.netloc:
        domain = parsed_url.netloc

        # Remove any leading "www."
        if domain.startswith("www."):
            domain = domain[4:]

        # Split the domain into parts
        domain_parts = domain.split(".")

        # Get the last two parts as the domain and extension
        cleaned_domain = ".".join(domain_parts[-2:])

        return cleaned_domain

    # Return the original string if it's not a valid URL
    return url


def _evaluate(
    preds: list[str], ground_truths: list[str], ground_truth_separator: str = " | "
) -> float:
    # lowercase
    preds = [pred.lower().strip() for pred in preds]
    ground_truths = [ground_truth.lower().strip() for ground_truth in ground_truths]
    res: list[int] = []
    for pred, ground_truth in zip(preds, ground_truths):
        ground_truth_options: list[str] = [
            gt.strip() for gt in ground_truth.split(ground_truth_separator)
        ]
        # clean urls (to relax url comparison)
        ground_truth_options = [gt for gt in ground_truth_options]
        pred = clean_url(pred)
        if pred in ground_truth_options:
            res.append(1)
        else:
            res.append(0)

    return sum(res) / len(res)


def evaluate(
    preds: list[dict],
    ground_truth_df: pd.DataFrame,
    correct_merchants_column: str = "correct_merchant",
    correct_websites_column: str = "correct_website",
):
    merchants_accuracy = _evaluate(
        preds=[pred["merchant"] for pred in preds],
        ground_truths=ground_truth_df[correct_merchants_column].to_list(),
    )
    websites_accuracy = _evaluate(
        preds=[pred["website"] for pred in preds],
        ground_truths=ground_truth_df[correct_websites_column].to_list(),
    )
    return {
        "Merchant accuracy": merchants_accuracy,
        "Website accuracy": websites_accuracy,
    }
