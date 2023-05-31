import functools
import os
import pickle
from pathlib import Path

import numpy as np
import openai
import pandas as pd  # type: ignore
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


def cache(file_path):
    def decorator(func):
        try:
            with open(file_path, "rb") as f:
                memo = pickle.load(f)
        except FileNotFoundError:
            memo = {}

        @functools.wraps(func)
        def wrapper(args):
            results = []
            for arg in args:
                if arg in memo:
                    results.append(memo[arg])
                else:
                    result = func([arg])
                    memo[arg] = result[0]
                    results.append(result[0])
            with open(file_path, "wb") as f:
                pickle.dump(memo, f)
            return results

        return wrapper

    return decorator


@cache(file_path=str(Path(__file__).parent / "openai_embeddings_cache.pickle"))
def get_openai_embeddings(
    texts: list[str],
    model="text-embedding-ada-002",
    prefix: str = "Bank transaction category: ",
    openai_api_key: str | None = None,
) -> list[np.ndarray]:
    texts = [text.replace("\n", " ") for text in texts]
    texts = [f"{prefix}{text}" for text in texts]
    if openai_api_key is None:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_key = openai_api_key
    res = openai.Embedding.create(input=texts, model=model)["data"]
    return [r["embedding"] for r in res]


def get_sentence_transformer_embeddings(
    texts: list[str],
    model="all-mpnet-base-v2",
    prefix: str = "Bank transaction category: ",
    normalize: bool = True,
) -> list[np.ndarray]:
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = [text.replace("\n", " ") for text in texts]
    texts = [f"{prefix}{text}" for text in texts]
    return model.encode(texts, normalize_embeddings=normalize)


def label_similarity_score(
    preds: list[str],
    ground_truths: list[str],
    normalize_distance: bool = True,
    max_normalizing_value: float | None = None,
    max_normalizing_quantile: float | None = 0.95,
    squared_similarity: bool = True,
    average_reduction: bool = True,
    max_similarity_allowed: float = 0.9,
    use_openai_model: bool = True,
    use_levenshtein: bool = True,
) -> float | list[float]:
    assert not (
        max_normalizing_quantile is not None and max_normalizing_value is not None
    ), "Cannot use max_normalized_value and max_normalized_quantile together"
    assert len(preds) == len(
        ground_truths
    ), "preds and ground truths must be the same size"
    unique_labels: list[str] = sorted(list(set(preds + ground_truths)))

    if use_openai_model:
        embeddings = get_openai_embeddings(unique_labels)
    else:
        embeddings = get_sentence_transformer_embeddings(unique_labels)
    distances_matrix = 1 - cosine_similarity(embeddings)
    np.fill_diagonal(distances_matrix, 0)

    if normalize_distance:
        # define a normalization function to be applied along each column
        def normalize_col(col: np.ndarray):
            min_val = np.min(col[np.nonzero(col)])
            if max_normalizing_value is not None:
                max_val = max_normalizing_value
            elif max_normalizing_quantile is not None:
                max_val = np.quantile(col, q=max_normalizing_quantile)
            else:
                max_val = col.max()
            return np.interp(col, (min_val, max_val), (0, 1))

        # apply the normalization function along each column
        distances_matrix = np.apply_along_axis(
            normalize_col, axis=0, arr=distances_matrix
        )

    np.fill_diagonal(distances_matrix, 0)
    similarities = []
    for pred, ground_truth in zip(preds, ground_truths):
        pred_index = unique_labels.index(pred)
        ground_truth_index = unique_labels.index(ground_truth)
        similarity = 1 - distances_matrix[pred_index, ground_truth_index]
        if use_levenshtein:
            levenshtein_similarity = Levenshtein.normalized_similarity(
                pred, ground_truth
            )
            if levenshtein_similarity > similarity:
                similarity = levenshtein_similarity
        if squared_similarity:
            similarity = similarity * similarity
        if pred != ground_truth and similarity > max_similarity_allowed:
            similarity = max_similarity_allowed
        similarities.append(similarity)

    if average_reduction:
        return sum(similarities) / len(similarities)
    return similarities


def _evaluate(
    preds: list[str], ground_truths: list[str], decimal_precision: int = 2
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, preds, average="macro", zero_division=0
    )
    accuracy = accuracy_score(ground_truths, preds)
    label_similarity: float | list[float] = label_similarity_score(
        preds, ground_truths, average_reduction=True
    )
    assert isinstance(label_similarity, float)
    return {
        "precision": round(precision, decimal_precision),
        "recall": round(recall, decimal_precision),
        "f1": round(f1, decimal_precision),
        "accuracy": round(accuracy, decimal_precision),
        "label_similarity": round(label_similarity, decimal_precision),
    }


def evaluate(
    preds: list[dict],
    ground_truth_df: pd.DataFrame,
    correct_labels_column: str = "correct_labels",
):
    score = _evaluate(
        preds=[pred["labels"] for pred in preds],
        ground_truths=ground_truth_df[correct_labels_column].to_list(),
    )
    return {f"Labeler {k}": v for k, v in score.items()}
