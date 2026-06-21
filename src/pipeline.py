from src.parsing import read_function, read_prompt, get_arguments
from src.parsing import chek_duplicated
import json
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
import time
from src.construct_json import generate_function_call
from typing import Any
import sys


def read_vocabulary_files(model: Small_LLM_Model) -> dict:
    """Read the vocabulary file and build a token ID to string mapping.

    Args:
        model: The language model providing the vocabulary file path.

    Returns:
        A dictionary mapping token IDs to their string representation.
    """

    path_vocabulary = model.get_path_to_vocab_file()
    vocabulary = {}
    with open(path_vocabulary, 'r') as vocabulary_files:
        dict_vocab = json.load(vocabulary_files)
        vocabulary = {k: v for v, k in dict_vocab.items()}
    return vocabulary


def delete_returns(_function: list) -> list:
    """Remove the "returns" field from function definitions.

    Args:
        _function: List of function definitions.

    Returns:
        A new list of function definitions without the "returns" field.
    """

    new_list = [{"name": x["name"], "description": x["description"],
                 "parameters": x["parameters"]} for x in _function]
    return new_list


def run_pipeline() -> None:
    """Run the full function-calling pipeline from input to output.

    Parses command-line arguments, reads and validates the input
    files, then generates a function call for every prompt using
    constrained decoding. Processing time is reported for each
    prompt and in total, and the results are written to the output
    JSON file.
    """

    function_path, prompt_path, output_path = get_arguments()
    _function = read_function(function_path)
    prompt = read_prompt(prompt_path)
    chek_duplicated(_function)
    results: list[dict[str, Any]] = []

    if not prompt:
        print("Warning: No prompts found in input file. No output file "
              "will be created.")
        sys.exit(1)

    if not _function:
        print("Error: functions_definition.json is empty or invalid. "
              "Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    model = Small_LLM_Model("Qwen/Qwen3-0.6B")
    _function = delete_returns(_function)
    vocabulary = read_vocabulary_files(model)
    name_function = [x['name'] for x in _function]

    total_time: float = 0
    for p in prompt:
        print(f"Prompt: \"{p['prompt']}\"")
        start = time.time()
        structure = generate_function_call(p, model, _function,
                                           vocabulary, name_function)
        print(f"Result: {json.dumps(structure)}")
        end = time.time()
        elapsed = end - start
        print(f'Elapsed: {elapsed:.2f} seconds.')
        print("---------")
        total_time += elapsed
        results.append(structure)
    total_time /= 60
    print(f"Total time: {total_time:.2f} min.")

    try:
        with open(output_path, 'w') as files:
            json.dump(results, files, indent=4)
    except OSError as e:
        print(f"Error: Cannot write output file: {e}", file=sys.stderr)
        sys.exit(1)
