from llm_sdk import Small_LLM_Model
import json
from src.utility_function import select_token_for_function_name
from src.utility_function import trim_function_names
from src.utility_function import select_token_by_type


def generate_function_call(prompt: dict, model: Small_LLM_Model,
                           _function: list, vocabulary: dict,
                           name_function: list) -> dict:
    structure = []
    prompt_text = json.dumps(prompt['prompt'])[1:-1]
    first_structure = f"{{\"prompt\": \"{prompt_text}\",\"name\": \""
    third_structure = "\",\"parameters\": "

    tensor = model.encode(f"Functions: {json.dumps(_function)}. "
                          f"Choose the BEST matching function "
                          "for this request. "
                          f"Request: {prompt_text}. "
                          f"Answer with the function name that best matches"
                          " the description.")
    input_ids = tensor[0].tolist()

    str_function_name = ""
    index_function = 0
    len_arguments = 0
    keys_and_types = []
    count_args = 0

    step = 1
    while True:
        if step == 1:
            tensor = model.encode(first_structure)
            first_structure = tensor[0].tolist()
            input_ids.extend(first_structure)
            structure.extend(first_structure)
            step += 1

        if step == 2:

            next_token = select_token_for_function_name(input_ids, vocabulary,
                                                        name_function, model)
            str_token = model.decode(next_token)

            str_function_name += str_token
            if (
                    len(name_function) == 1
                    and name_function[0].endswith(str_token)
                    ):
                structure.append(int(next_token))
                input_ids.append(int(next_token))
                step += 1
                continue
            else:
                structure.append(int(next_token))
                input_ids.append(int(next_token))

            name_function = [x for x in name_function
                             if x.startswith(str_token)]
            name_function = trim_function_names(len(str_token), name_function)
            if len(name_function) == 1 and not name_function[0]:
                step += 1

        if step == 3:
            tensor = model.encode(third_structure)
            third_structure = tensor[0].tolist()
            input_ids.extend(third_structure)
            structure.extend(third_structure)

            for d in _function:
                if d['name'] == str_function_name:
                    break
                index_function += 1
            for k, v in _function[index_function]['parameters'].items():
                keys_and_types.append((k, v['type']))
                len_arguments += 1

            step += 1

        if step == 4:
            if count_args == len_arguments:
                tensor = model.encode("}")
                braces = tensor[0].tolist()
                input_ids.extend(braces)
                structure.extend(braces)
                input_ids.extend(braces)
                structure.extend(braces)
                break

            key_arg = f"{{\"{keys_and_types[count_args][0]}\": "
            if count_args > 0:
                key_arg = f"\"{keys_and_types[count_args][0]}\": "
            key_or_value = "key"
            type_arg = keys_and_types[count_args][1]

            while True:

                if key_or_value == "key":
                    tensor = model.encode(key_arg)
                    key_arg = tensor[0].tolist()
                    input_ids.extend(key_arg)
                    structure.extend(key_arg)
                    key_or_value = "value"

                if key_or_value == "value":

                    if type_arg == "integer":
                        next_token = select_token_by_type(
                            input_ids, vocabulary,
                            model, "number", tuple("0123456789")
                            )
                        decode_token = model.decode(int(next_token))
                        count_zero = 0
                        while decode_token.startswith(tuple("0123456789")):
                            if decode_token.startswith('0'):
                                count_zero += 1
                            if count_zero > 1:
                                break
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            next_token = select_token_by_type(
                                input_ids, vocabulary,
                                model, "number", tuple("0123456789},")
                                )
                            decode_token = model.decode(next_token)
                        count_args += 1

                    if type_arg == 'number':
                        next_token = select_token_by_type(
                            input_ids, vocabulary,
                            model, "number", tuple("0123456789")
                            )
                        decode_token = model.decode(int(next_token))
                        count_zero = 0
                        while decode_token.startswith(tuple("0123456789.")):
                            if decode_token.startswith('0'):
                                count_zero += 1
                            if count_zero > 1:
                                break
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            next_token = select_token_by_type(
                                input_ids, vocabulary,
                                model, "number", tuple("0123456789.},")
                                )
                            decode_token = model.decode(next_token)
                        count_args += 1

                    if type_arg == 'string':
                        next_token = select_token_by_type(
                            input_ids, vocabulary, model, "quotes"
                            )
                        input_ids.append(int(next_token))
                        structure.append(int(next_token))
                        decode_token = model.decode(int(next_token))
                        while True:
                            next_token = select_token_by_type(
                                input_ids, vocabulary, model, "string"
                                )
                            decode_token = model.decode(int(next_token))
                            if decode_token.endswith('"'):
                                next_token = select_token_by_type(
                                    input_ids, vocabulary, model, "quotes"
                                    )
                                input_ids.append(int(next_token))
                                structure.append(int(next_token))
                                break
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                        count_args += 1

                    if type_arg == 'boolean':
                        next_token = select_token_by_type(
                            input_ids, vocabulary, model, "boolean"
                            )
                        input_ids.append(int(next_token))
                        structure.append(int(next_token))
                        count_args += 1

                    if type_arg == 'none':
                        tensor = model.encode("null")
                        _null = tensor[0].tolist()
                        input_ids.extend(_null)
                        structure.extend(_null)
                        count_args += 1

                    if count_args < len_arguments:
                        tensor = model.encode(",")
                        commas = tensor[0].tolist()
                        input_ids.extend(commas)
                        structure.extend(commas)
                    break
    dict_ = json.loads(model.decode(structure))
    return dict_
