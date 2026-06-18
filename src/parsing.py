import argparse as agp
import json
from src.models import FunctionModel, PromptModel
from pydantic import ValidationError
import sys
from pathlib import Path
from typing import Any


def get_arguments() -> tuple[str, str, str]:
    """Parse command-line arguments and ensure the output directory exists.

    Returns:
        A tuple containing the paths to the functions definition,
        input, and output files.
    """

    parser = agp.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calls.json")
    args = parser.parse_args()
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Cannot create output directory: {e}", file=sys.stderr)
        sys.exit(1)
    return args.functions_definition, args.input, args.output


def read_function(path: str) -> list[dict[str, Any]]:
    """Read and validate function definitions from a JSON file.

    Each function definition is validated against the expected schema.
    The program exits if the file is missing, contains invalid JSON,
    or if any function definition fails validation.

    Args:
        path: Path to the functions definition file.

    Returns:
        A list of raw function definitions as dictionaries.
    """

    data_json: list[dict] = []
    try:
        with open(path, 'r') as func_file:
            try:
                data_json = json.load(func_file)
            except json.JSONDecodeError as e:
                print(f"Error: {path} must be valid Json no trailing commas,"
                      " no comments\n"
                      f"{e}", file=sys.stderr)
                sys.exit(1)
            for data in data_json:
                try:
                    FunctionModel(**data)
                except ValidationError as e:
                    for error in e.errors():
                        print(f"Error in function_definition: type "
                              f"{error['type']}"
                              f" loc {error['loc']} {error['msg']}\nInput: "
                              f"{error['input']}", file=sys.stderr)
                    sys.exit(1)
    except OSError as e:
        print(f"Error: Cannot read file {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return data_json


def read_prompt(path: str) -> list[dict[str, Any]]:
    """Read and validate prompts from a JSON file.

    Each prompt is validated against the expected schema. Invalid
    prompts are skipped with a warning, while valid ones are kept.

    Args:
        path: Path to the prompts file.

    Returns:
        A list of valid prompt dictionaries.
    """

    try:
        with open(path, 'r') as prompt_file:
            try:
                data_json = json.load(prompt_file)
            except json.JSONDecodeError as e:
                print(f"Error: {path} must be valid Json no trailing commas,"
                      " no comments\n"
                      f"{e}", file=sys.stderr)
                sys.exit(1)
            data_lst: list[dict] = []
            for data in data_json:
                try:
                    PromptModel(**data)
                    data_lst.append(data)
                except ValidationError:
                    print(f"Warning: Invalid prompt ignored : {data}",
                          file=sys.stderr)
                    continue
    except OSError as e:
        print(f"Error: Cannot read file {path}: {e}", file=sys.stderr)
        sys.exit(1)
    return data_lst
