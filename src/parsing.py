import argparse as agp
import json
from src.models import FunctionModel, PromptModel
from pydantic import ValidationError
import sys


def get_arguments() -> list:
    parse = agp.ArgumentParser()
    parse.add_argument("--functions_definition",
                       default="data/input/functions_definition.json")
    parse.add_argument("--input",
                       default="data/input/function_calling_tests.json")
    parse.add_argument("--output", default="data/output/function_calls.json")
    args = parse.parse_args()
    return [args.functions_definition,
            args.input, args.output]


def read_function(path: str) -> list:
    try:
        with open(path, 'r') as func_file:
            try:
                data_json = json.load(func_file)
            except json.JSONDecodeError as e:
                print("Error: The files must be valid Json no trailing commas,"
                      " no comments\n"
                      f"{e}")
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
    except FileNotFoundError:
        print(f"Error: file {path} not found", file=sys.stderr)
        sys.exit(1)
    return data_json


def read_prompt(path: str) -> list:
    data_lst = []
    try:
        with open(path, 'r') as prompt_file:
            try:
                data_json = json.load(prompt_file)
            except json.JSONDecodeError as e:
                print("Error: The files must be valid Json no trailing commas,"
                      " no comments\n"
                      f"{e}")
                return []
            data_lst = []
            for data in data_json:
                try:
                    PromptModel(**data)
                    data_lst.append(data)
                except ValidationError:
                    print(f"Warning: Invalid prompt ignored : {data}",
                          file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: file {path} not found", file=sys.stderr)
        sys.exit(1)
    return data_lst
