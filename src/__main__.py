from src.parsing import read_function, read_prompt, get_arguments
import json
from llm_sdk import Small_LLM_Model
import time
from src.construct_json import generate_function_call


def generate_all_function_call():
    path = get_arguments()
    prompt = read_prompt(path[1])
    _function = read_function(path[0])
    _result = []
    if not prompt:
        print("Warning: No prompts found in input file. Output will be empty.")
        return _result
    if not _function:
        raise ValueError("Error: functions_definition.json is empty or invalid"
                         ". Cannot proceed.")
    model = Small_LLM_Model("Qwen/Qwen3-0.6B")
    path_vocab = model.get_path_to_vocab_file()

    vocabulary = {}
    name_function = [x['name'] for x in _function]
    with open(path_vocab, 'r') as vocabulary_files:
        dict_vocab = json.load(vocabulary_files)
        vocabulary = {k: v for v, k in dict_vocab.items()}

    my_lst = []
    for x in _function:
        my_dict = {}
        my_dict['name'] = x['name']
        my_dict['description'] = x['description']
        my_dict['parameters'] = x['parameters']
        my_lst.append(my_dict)
    _function = my_lst
    total_time = 0
    for p in prompt:
        print(f"Prompt: \"{p['prompt']}\"")
        start = time.time()
        structure = generate_function_call(p, model, _function,
                                           vocabulary, name_function)
        end = time.time()
        elapsed = end - start
        print(f'Elapsed: {elapsed:.2f} seconds')
        print("---------")
        total_time += elapsed
        _result.append(structure)
    total_time /= 60
    print(f"Total time: {total_time:.2f}min")
    return _result


def output_function_call(_result, path):
    with open(path[2], 'w') as files:
        json.dump(_result, files, indent=4)


output_function_call(generate_all_function_call(), get_arguments())
