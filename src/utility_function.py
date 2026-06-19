import numpy as np
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from typing import Optional


def select_token_for_function_name(input_ids: list,
                                   vocabulary: dict, name_function: list,
                                   model: Small_LLM_Model) -> int:
    """Select the next token constrained to valid function name prefixes.

    All token logits are masked to negative infinity except those
    whose string representation is a valid prefix of at least one
    candidate function name. The token with the highest logit among
    the valid candidates is selected.

    Args:
        input_ids: List of token IDs representing the current prompt.
        vocabulary: Mapping of token IDs to their string representation.
        name_function: List of candidate function names.
        model: The language model used to generate logits.

    Returns:
        The ID of the selected token.
    """

    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = []
    for name in name_function:
        ids = [token_id for token_id, token_str in vocabulary.items()
               if name.startswith(token_str) or token_str.endswith('"')]
        valid_ids.extend(ids)
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return int(next_token)


def trim_function_names(len_token: int, name_function: list) -> list:
    """Remove the already-generated prefix from candidate function names.

    Args:
        len_token: Length of the prefix already generated.
        name_function: List of candidate function names.

    Returns:
        The list of candidate names with the prefix removed.
    """

    return [name[len_token:] for name in name_function
            if len(name) >= len_token]


def select_token_by_type(input_ids: list, vocabulary: dict,
                         model: Small_LLM_Model, arg_type: str,
                         data: Optional[tuple] = None) -> int:
    """Select the next token constrained to a given argument type.

    A list of valid token IDs is built according to the argument
    type. All token logits are masked to negative infinity except
    those belonging to the valid set. The token with the highest
    logit among the valid candidates is selected.

    Args:
        input_ids: List of token IDs representing the current prompt.
        vocabulary: Mapping of token IDs to their string representation.
        model: The language model used to generate logits.
        arg_type: The expected argument type, one of "number",
        "integer", "string", "quotes", or "boolean".
        data: Required for "number" and "integer" types; contains
            the numeric prefix already generated.

    Returns:
        The ID of the selected token.

    Raises:
        ValueError: If arg_type is not one of the supported types.
    """

    valid_ids = []
    if arg_type == "number" or arg_type == "integer":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if token_str.startswith(data)]
    elif arg_type == "string":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if ('"' not in token_str or token_str.endswith('"'))
                     ]
    elif arg_type == "quotes":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if token_str.startswith('"') and len(token_str) == 1]
    elif arg_type == "boolean":
        valid_ids = [token_id for token_id, token_str in vocabulary.items()
                     if token_str.startswith('true')
                     or token_str.startswith('false')]
    else:
        raise ValueError(f"Unsupported argument type: {arg_type}")

    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return int(next_token)
