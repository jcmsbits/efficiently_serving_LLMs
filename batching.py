import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")

def generate_token_with_past(inputs):    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    last_logits = logits[0,-1, :]
    next_token_id = last_logits.argmax()
    return next_token_id, outputs.past_key_values


def generate(inputs, max_tokens):
    print("Generando....")
    generated_tokens = []
    next_inputs = inputs
    durations_cached_s = []

    for _ in range(max_tokens):
        next_token_id, past_key_values = \
            generate_token_with_past(next_inputs)
        next_inputs = {
            "input_ids" : next_token_id.reshape((-1,1)),
            "attention_mask" : torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],
                dim = 1
            ),
            "past_key_values" : past_key_values,
        }
        next_token = tokenizer.decode(next_token_id)
        print("Token generado:", next_token)
        generated_tokens.append(next_token)

    return "".join(generated_tokens)

tokens = generate(inputs, max_tokens=10)
print(tokens)

# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# pad on the left so we can  append new tokens on the right
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

# multiple prompts of varying lengths to send
# to the model at once 

prompts = [
    "The quick brown fox jumped over the",
    "The rain in Spain falls",
    "What comes up must"
]

# note: padding = True ensure padding token
# will be inserted into the tokenized tensors

inputs = tokenizer(prompts, padding = True, return_tensors="pt")

print("input_ids:", inputs["input_ids"])
print("shape", inputs["input_ids"].shape)

print("attention_mask", inputs["attention_mask"])
print("shape", inputs["attention_mask"].shape)

# position_ids tell the transformer the ordinal position
# of each token in the input sequence
# for single input inference, this is just [0 .. n]
# for n tokens, but for batch inference,
# we need to 0 out the padding tokens at the start of the sequence

attention_mask = inputs["attention_mask"]
position_ids = attention_mask.long().cumsum(-1) - 1
print("Position ids without masked:", position_ids)
position_ids_masked = position_ids.masked_fill(attention_mask == 0, 1 )
print("Position ids with masked_fill:", position_ids_masked)

# Same as before, but include the position_ids
with torch.no_grad():
    outputs = model(position_ids = position_ids_masked, **inputs)

logits = outputs.logits

# Obtenemos el token con mayor probabilidad pero en todos los lotes
last_logits = logits[:, -1, :]
next_token_ids = last_logits.argmax(dim = 1)

print(next_token_ids)

next_tokens = tokenizer.batch_decode(next_token_ids)
print(next_tokens)


def generate_batch_tokens_with_past(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    last_logits = logits[:, -1 ,:]
    next_tokens_ids = last_logits.argmax(dim = 1)
    return next_tokens_ids, outputs.past_key_values

def generate_batch(inputs, max_tokens):
    # create a list of tokens for every input in the batch
    shape_input_ids = inputs["input_ids"].shape[0]
    print("Shape input ids:", shape_input_ids)
    generated_tokens = [
        [] for _ in range(shape_input_ids)
    ]
    print("Pase el for...")
    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    masked_position = position_ids.masked_fill(attention_mask == 0, 1)
    print("Pase el masking...")
    next_inputs = {
        "position_ids" : masked_position,
        **inputs
    }

    for _ in range(max_tokens):
        next_token_ids, past_key_values = \
            generate_batch_tokens_with_past(next_inputs)
        print("Empece a generar tokens......")
        print("Position ids sin squeeze:", next_inputs["position_ids"])
        position_ids_unqueeze = next_inputs["position_ids"][:,-1].unsqueeze(-1) + 1
        print("Tokens generados en batch:", next_token_ids)
        print("Position id unqueeze: ", position_ids_unqueeze)

        reshaped_tokens = next_token_ids.reshape((-1,1))
        print("Reshape tokens: ", reshaped_tokens)

        print("Next token ids:", reshaped_tokens.shape)
        ones = torch.ones((reshaped_tokens.shape[0], 1))

        print("Ones: ", ones)
        next_inputs = {
            "input_ids" : next_token_ids.reshape((-1,1)),
            "position_ids" : position_ids_unqueeze,
            "attention_mask" : torch.cat([
                next_inputs["attention_mask"],
                ones,
                ],
                dim = 1
            ),
            "past_key_values": past_key_values,
        }
        # print("Past Key Values: ",past_key_values )
        print("Next token ids: ", next_token_ids)
        next_tokens = tokenizer.batch_decode(next_token_ids)
        print("Next tokens: ", next_tokens)
        for i, token in enumerate(next_tokens):
            generated_tokens[i].append(token)
    print("Generated tokens: ", generated_tokens)
    returned = ["".join(tokens) for tokens in generated_tokens]
    print("Return: ", returned)
    return returned

generated_tokens = generate_batch(inputs, max_tokens=10)