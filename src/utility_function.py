import numpy as np
from llm_sdk import Small_LLM_Model


def select_token_for_function_name(input_ids: list,
                                   vocabulary: dict, name_function: list,
                                   model: Small_LLM_Model) -> int:
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = []
    for name in name_function:
        ids = [token_id for token_id, token_str in vocabulary.items()
               if name.startswith(token_str)]
        valid_ids.extend(ids)
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return int(next_token)


def trim_function_names(len_token: int, name_function: list) -> list:
    return [name[len_token:] for name in name_function
            if len(name) >= len_token]


def select_token_by_type(input_ids: list, vocabulary: dict,
                         model: Small_LLM_Model, arg_type: str,
                         data=None) -> int:
    valid_ids = []
    if arg_type == "number" or arg_type == "integer":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if token_str.startswith(data)]
    if arg_type == "string":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if '"' not in token_str or token_str.endswith('"')]
    if arg_type == "quotes":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if token_str.startswith('"') and len(token_str) == 1]
    if arg_type == "boolean":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if token_str.startswith('true')
                     or token_str.startswith('false')]

    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return int(next_token)
