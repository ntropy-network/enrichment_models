import time
from pathlib import Path

import pandas as pd  # type: ignore

from enrichment_models.llms.llama import LLAMA
from enrichment_models.llms.openai import ChatGPT
from enrichment_models.tasks.labeler.ntropy_wrapper import NtropyLabeler
from enrichment_models.tasks.labeler.prompter import LabelerPrompter
from enrichment_models.tasks.labeler.utils import evaluate as evaluate_labeler
from enrichment_models.tasks.llm_wrapper import LLMWrapper
from enrichment_models.tasks.normalizer.ntropy_wrapper import NtropyNormalizer
from enrichment_models.tasks.normalizer.prompter import NormalizerPrompter
from enrichment_models.tasks.normalizer.utils import evaluate as evaluate_normalizer

TEST_SET_PATH = (
    Path(__file__).parent.parent / "datasets/100_labeled_consumer_transactions.csv"
)
OUTPUT_PATH_METRICS = Path(__file__).parent.parent / "datasets/benchmark_results.csv"
OUTPUT_PATH_PREDICTIONS = (
    Path(__file__).parent.parent / "datasets/benchmark_predictions.csv"
)
GROUND_TRUTH_LABELS_COLUMN = "labels_correct"
GROUND_TRUTH_MERCHANT_COLUMN = "merchant_correct"
GROUND_TRUTH_WEBSITE_COLUMN = "website_correct"


# Loading dataset
test_set_df = pd.read_csv(TEST_SET_PATH)

# Note: we init the models during the for loop only, to preserve RAM (because of LLAMA's models)
LABELER_MODELS: dict[str, str] = {
    # LABELER TASK
    "GPT 3": """LLMWrapper(
        llm=ChatGPT(model_name="gpt-3.5-turbo"),
        prompter=LabelerPrompter(),
    )""",
    "GPT 4": """LLMWrapper(
        llm=ChatGPT(model_name="gpt-4"),
        prompter=LabelerPrompter(),
    )""",
    "LLAMA finetuned 7B": """LLMWrapper(
        llm=LLAMA(
            base_model="decapoda-research/llama-7b-hf",
            lora_weights="ntropydev/ntropy-labeler-llama-lora-7b",
            batch_size=16,
        ),
        prompter=LabelerPrompter(),
    )""",
    "LLAMA finetuned 13B": """LLMWrapper(
        llm=LLAMA(
            base_model="decapoda-research/llama-13b-hf",
            lora_weights="ntropydev/ntropy-labeler-llama-lora-13b",
            batch_size=16,
        ),
        prompter=LabelerPrompter(),
    )""",
    "Ntropy API": """NtropyLabeler()""",
}

NORMALIZER_MODELS: dict[str, str] = {
    "GPT 3": """LLMWrapper(
        llm=ChatGPT(model_name="gpt-3.5-turbo"),
        prompter=NormalizerPrompter(),
    )""",
    "GPT 4": """LLMWrapper(
        llm=ChatGPT(model_name="gpt-4"),
        prompter=NormalizerPrompter(),
    )""",
    "Ntropy API": """NtropyNormalizer()""",
}

# Init results
evaluation_scores = pd.DataFrame()

# Evaluate labeler models
for model_key, model_init_str in LABELER_MODELS.items():
    print(f"Inference on Labeler task using {model_key}")
    labeler_model: LLMWrapper | NtropyLabeler = eval(model_init_str)
    start_time = time.time()
    preds = labeler_model.predict(test_set_df)
    end_time = time.time()
    aprox_time_per_tx = round((end_time - start_time) / len(test_set_df), 2)
    score = evaluate_labeler(
        preds, test_set_df, correct_labels_column=GROUND_TRUTH_LABELS_COLUMN
    )

    evaluation_scores.loc["Labeler Accuracy", model_key] = score["Labeler accuracy"]
    evaluation_scores.loc["Labeler F1 Score", model_key] = score["Labeler f1"]
    evaluation_scores.loc["Labeler Label similarity", model_key] = score[
        "Labeler label_similarity"
    ]
    evaluation_scores.loc["Labeler latency (s/tx)", model_key] = aprox_time_per_tx

    test_set_df[f"prediction_labels_{model_key}"] = [pred["labels"] for pred in preds]

# Evaluate normalizer models
for model_key, model_init_str in NORMALIZER_MODELS.items():
    print(f"Inference on Normalizer task using {model_key}")
    normalizer_model: LLMWrapper | NtropyNormalizer = eval(model_init_str)
    start_time = time.time()
    preds = normalizer_model.predict(test_set_df)
    end_time = time.time()
    aprox_time_per_tx = round((end_time - start_time) / len(test_set_df), 2)
    score = evaluate_normalizer(
        preds,
        test_set_df,
        correct_merchants_column=GROUND_TRUTH_MERCHANT_COLUMN,
        correct_websites_column=GROUND_TRUTH_WEBSITE_COLUMN,
    )

    evaluation_scores.loc["Merchant Accuracy", model_key] = score["Merchant accuracy"]
    evaluation_scores.loc["Website Accuracy", model_key] = score["Website accuracy"]
    evaluation_scores.loc["Normalizer latency (s/tx)", model_key] = aprox_time_per_tx

    test_set_df[f"prediction_merchant_{model_key}"] = [
        pred["merchant"] for pred in preds
    ]
    test_set_df[f"prediction_website_{model_key}"] = [pred["website"] for pred in preds]


# Show benchmark results
print(evaluation_scores)
evaluation_scores.to_csv(OUTPUT_PATH_METRICS)
test_set_df.to_csv(OUTPUT_PATH_PREDICTIONS, index=False)
