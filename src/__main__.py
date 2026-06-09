from src.parsing import read_function, read_prompt, get_arguments
import json
from llm_sdk import Small_LLM_Model
import numpy as np


def name_and_decription(_function: list) -> list:
    result = []

    for func in _function:
        _str = f"{func['name']}: {func['description']}"
        result.append(_str)

    return result


def structure_fix(input_ids: list, structure_one: str, vocabulary: dict, model) -> int:
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = [token_id for token_id, token_str in vocabulary.items()
                 if structure_one.startswith(token_str)]
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return next_token


def function_structure(input_ids: list, vocabulary: dict, name_function: list, model) -> int:
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = []
    for name in name_function:
        ids = [token_id for token_id, token_str in vocabulary.items()
               if name.startswith(token_str)]
        valid_ids.extend(ids)
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return next_token


def new_name_function(len_token: int, name_function: list) -> list:
    return [name[len_token:] for name in name_function if len(name) >= len_token]


def type_number(input_ids: list, vocabulary: dict, model):
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith(tuple("0123456789-.},"))]
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return next_token


def type_string(input_ids: list, vocabulary: dict, model):
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = valid_ids = [token_id for token_id, token_str in vocabulary.items() if '"' not in token_str or token_str.endswith('"')]
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return next_token


def main_loop(prompt, model, _function, vocabulary, name_function):
    structure = []
    prompt_text = json.dumps(prompt['prompt'])[1:-1]
    first_structure = f"{{\"prompt\": \"{prompt_text}\",\"name\": \""
    first_structure = first_structure.replace(' ', 'Ġ')

    third_structure = "\",\"parameters\": "
    third_structure = third_structure.replace(' ', 'Ġ')

    prompt_json_str = json.dumps(prompt)
    tensor = model.encode(f"{json.dumps(_function)} Give name of description solve this: {prompt_json_str}.")
    input_ids = tensor[0].tolist()
    step = 1

    str_function_name = ""

    index_function = 0
    len_arguments = 0
    key_arguments = []
    types_arguments = []
    i = 0

    while True:
        if step == 1:
            next_token = structure_fix(input_ids, first_structure,
                                       vocabulary, model)
            structure.append(int(next_token))
            input_ids.append(int(next_token))

            decode_next_token = model.decode(int(next_token))
            len_next_token = len(decode_next_token)
            first_structure = first_structure[len_next_token:]
            if not first_structure:
                step += 1

        if step == 2:

            next_token = function_structure(input_ids, vocabulary,
                                            name_function, model)
            str_token = model.decode(next_token)

            str_function_name += str_token
            if len(name_function) == 1 and name_function[0].endswith(str_token):
                structure.append(int(next_token))
                input_ids.append(int(next_token))
                step += 1
                continue
            else:
                structure.append(int(next_token))
                input_ids.append(int(next_token))

            name_function = [x for x in name_function
                             if x.startswith(str_token)]
            name_function = new_name_function(len(str_token), name_function)
            if len(name_function) == 1 and not name_function[0]:
                step += 1

        if step == 3:
            next_token = structure_fix(input_ids, third_structure, vocabulary, model)

            structure.append(int(next_token))
            input_ids.append(int(next_token))

            decode_next_token = model.decode(int(next_token))
            len_next_token = len(decode_next_token)

            third_structure = third_structure[len_next_token:]

            if not third_structure:

                for d in _function:
                    if d['name'] == str_function_name:
                        break
                    index_function += 1
                for k, v in _function[index_function]['parameters'].items():
                    key_arguments.append(k)
                    types_arguments.append(v['type'])

                    len_arguments += 1
                step += 1

        if step == 4:
            if i == len_arguments:
                break
            key_argument = f"{{\"{key_arguments[i]}\":Ġ"
            if i > 0:
                key_argument = f"\"{key_arguments[i]}\":Ġ"
            key_or_value = "key"
            type_arg = types_arguments[i]

            while True:

                if key_or_value == "key":
                    next_token = structure_fix(input_ids, key_argument,
                                               vocabulary, model)

                    structure.append(int(next_token))
                    input_ids.append(int(next_token))

                    decode_next_token = model.decode(int(next_token))
                    len_next_token = len(decode_next_token)

                    key_argument = key_argument[len_next_token:]

                    if not key_argument:
                        key_or_value = "value"

                if key_or_value == "value":
                    if type_arg == 'number':
                        next_token = type_number(input_ids, vocabulary, model)
                        decode_next_token = model.decode(int(next_token))
                        while decode_next_token.startswith(tuple("0123456789-.")):
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            next_token = type_number(input_ids, vocabulary, model)
                            decode_next_token = model.decode(int(next_token))

                        i += 1
                        if i < len_arguments:
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith(",") and len(token_str) == 1]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))

                        if i == len_arguments:
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith("}") and len(token_str) == 1]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith("}") and len(token_str) == 1]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))

                    if type_arg == 'string':
                        logits = np.array(model.get_logits_from_input_ids(input_ids))
                        mask = np.full(len(logits), float('-inf'))
                        valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith('"') and len(token_str) == 1]
                        mask[valid_ids] = logits[valid_ids]
                        next_token = np.argmax(mask)
                        input_ids.append(int(next_token))
                        structure.append(int(next_token))
                        decode_next_token = model.decode(int(next_token))
                        while True:
                            next_token = type_string(input_ids, vocabulary, model)
                            decode_next_token = model.decode(int(next_token))
                            if decode_next_token.endswith('"'):
                                logits = np.array(model.get_logits_from_input_ids(input_ids))
                                mask = np.full(len(logits), float('-inf'))
                                valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith('"') and len(token_str) == 1]
                                mask[valid_ids] = logits[valid_ids]
                                next_token = np.argmax(mask)
                                input_ids.append(int(next_token))
                                structure.append(int(next_token))
                                break
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                        i += 1
                        if i < len_arguments:
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith(",") and len(token_str) == 1]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            decode_next_token = model.decode(int(next_token))
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))

                        if i == len_arguments:
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith("}") and len(token_str) == 1]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith("}") and len(token_str) == 1]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                    break
    dict_ = json.loads(model.decode(structure))
    return dict_


def result():
    model = Small_LLM_Model("Qwen/Qwen3-0.6B")
    path = get_arguments()
    prompt = read_prompt(path[1])
    _function = read_function(path[0])
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
    _result = []
    for p in prompt:
        print(p)
        structure = main_loop(p, model, _function, vocabulary, name_function)
        _result.append(structure)

    with open('output.txt', 'w') as files:
        files.write(str(_result))

    with open('output.json', 'w') as files:
        json.dump(_result, files, indent=4)


result()
