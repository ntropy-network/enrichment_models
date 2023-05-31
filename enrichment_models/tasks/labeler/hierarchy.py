import requests  # type: ignore

CONSUMER_HIERARCHY_URL = "https://api.ntropy.com/v2/labels/hierarchy/consumer"


def get_consumer_hierarchy_ntropy(url: str = CONSUMER_HIERARCHY_URL) -> dict:
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            f"Response from Ntropy consumer hierarchy not valid: \n{response}"
        )
    consumer_hierarchy = response.json()
    return consumer_hierarchy


def get_consumer_labels_debit_ntropy(consumer_hierarchy: dict) -> list[str]:
    outgoing_groups = [
        "Essential Expenses",
        "Non-Essential Expenses",
        "Other Outgoing Transactions",
    ]
    debit_labels = []
    for group, labels in consumer_hierarchy.items():
        if group in outgoing_groups:
            debit_labels.extend(labels)
    label_to_remove = "missing account holder information"
    if label_to_remove in debit_labels:
        debit_labels.remove(label_to_remove)
    return sorted(debit_labels)


def get_consumer_labels_credit_ntropy(consumer_hierarchy: dict) -> list[str]:
    incoming_groups = [
        "Earned Income",
        "Passive Income",
        "Other Incoming Transactions",
    ]
    credit_labels = []
    for group, labels in consumer_hierarchy.items():
        if group in incoming_groups:
            credit_labels.extend(labels)
    label_to_remove = "missing account holder information"
    if label_to_remove in credit_labels:
        credit_labels.remove(label_to_remove)
    return sorted(credit_labels)


CONSUMER_HIERARCHY = get_consumer_hierarchy_ntropy(CONSUMER_HIERARCHY_URL)
CONSUMER_LABELS_DEBIT = get_consumer_labels_debit_ntropy(CONSUMER_HIERARCHY)
CONSUMER_LABELS_CREDIT = get_consumer_labels_credit_ntropy(CONSUMER_HIERARCHY)
NOT_ENOUGH_INFO_LABEL = "Not enough information"
