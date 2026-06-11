from llm_sdk import Small_LLM_Model
import numpy as np

model = Small_LLM_Model("Qwen/Qwen3-0.6B")
tensor = model.encode("What is the sum of -2 and -3?")
input_ids = tensor[0].tolist()
logits = model.get_logits_from_input_ids(input_ids)
token = np.argmax(logits)
print(model.decode(token))
input_ids.append(token)
logits = model.get_logits_from_input_ids(input_ids)
token = np.argmax(logits)
print(model.decode(token))
input_ids.append(token)
logits = model.get_logits_from_input_ids(input_ids)
token = np.argmax(logits)
print(model.decode(token))