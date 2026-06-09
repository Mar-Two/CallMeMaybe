# function_for_model = []
#    for f in func:
#        str_f = ""
#        str_f += f['name']
#        str_f += ': '
#        str_f += f['description']
#        function_for_model.append(str_f)
# tokens = model.encode(f"here is a list of json {func_str} give me the name of the description that solves that Reverse the string 'hello'")
#    lst = tokens[0].tolist()
#    count = 0
#    result = []
#    while True:
#        my_dct = {}
#        max_id = 0
#        for i, x in enumerate(model.get_logits_from_input_ids(lst)):
#            if x > max_id:
#                max_id = x
#                my_dct[max_id] = i
#        lst.append(my_dct[max_id])
#        result.append(my_dct[max_id])
#        count += 1
#        if count == 20:
#            break
#    print(model.decode(result))
def type_bool(input_ids: list, vocabulary: dict):
    logits = np.array(model.get_logits_from_input_ids(input_ids))
    mask = np.full(len(logits), float('-inf'))
    valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith('t', 'f')]
    mask[valid_ids] = logits[valid_ids]
    next_token = np.argmax(mask)
    return next_token

if type_arg == "bool":
                        next_token = type_bool(input_ids, vocabulary)
                        structure.append(int(next_token))
                        input_ids.append(int(next_token))

                        argument_tokens.append(next_token)
                        str_argument = model.decode(argument_tokens)
                        if str_argument == 'true' or str_argument == 'false':
                            break

if key_or_value == "value":
                    if type_arg == 'number':
                        next_token = type_number(input_ids, vocabulary)
                        input_ids.append(int(next_token))
                        structure.append(int(next_token))

                        len_token = len(model.decode(next_token))

                        while not actual_prompt.startswith(model.decode(next_token)):
                            if not actual_prompt:
                                break
                            actual_prompt = actual_prompt[1:]

                        actual_prompt = actual_prompt[len_token:]
                        if not actual_prompt.startswith(tuple("0123456789-")):
                            i += 1
                            break

if i < len_arguments:
                            logits = np.array(model.get_logits_from_input_ids(input_ids))
                            mask = np.full(len(logits), float('-inf'))
                            valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith(",")]
                            mask[valid_ids] = logits[valid_ids]
                            next_token = np.argmax(mask)
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))

def loopmain():
        while True:
        if step == 1:
            next_token = structure_fix(input_ids, first_structure,
                                       vocabulary)
            structure.append(int(next_token))
            input_ids.append(int(next_token))

            decode_next_token = model.decode(int(next_token))
            len_next_token = len(decode_next_token)
            first_structure = first_structure[len_next_token:]
            if not first_structure:
                step += 1

        if step == 2:

            next_token = function_structure(input_ids, vocabulary,
                                            name_function)
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
            next_token = structure_fix(input_ids, third_structure, vocabulary)

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
                                               vocabulary)

                    structure.append(int(next_token))
                    input_ids.append(int(next_token))

                    decode_next_token = model.decode(int(next_token))
                    len_next_token = len(decode_next_token)

                    key_argument = key_argument[len_next_token:]

                    if not key_argument:
                        key_or_value = "value"

                if key_or_value == "value":
                    if type_arg == 'number':
                        next_token = type_number(input_ids, vocabulary)
                        decode_next_token = model.decode(int(next_token))
                        while decode_next_token.startswith(tuple("0123456789-.")):
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            next_token = type_number(input_ids, vocabulary)
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
                        valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith('"') and token_str.endswith('"') and len(token_str) == 1]
                        mask[valid_ids] = logits[valid_ids]
                        next_token = np.argmax(mask)
                        decode_next_token = model.decode(int(next_token))

                        while True:
                            input_ids.append(int(next_token))
                            structure.append(int(next_token))
                            next_token = type_string(input_ids, vocabulary)
                            decode_next_token = model.decode(int(next_token))
                            if decode_next_token.endswith('"'):
                                input_ids.append(int(next_token))
                                structure.append(int(next_token))
                                break

                            if decode_next_token.startswith('"'):
                                logits = np.array(model.get_logits_from_input_ids(input_ids))
                                mask = np.full(len(logits), float('-inf'))
                                valid_ids = [token_id for token_id, token_str in vocabulary.items() if token_str.startswith('"') and token_str.endswith('"') and len(token_str) == 1]
                                mask[valid_ids] = logits[valid_ids]
                                next_token = np.argmax(mask)
                                input_ids.append(int(next_token))
                                structure.append(int(next_token))
                                break
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