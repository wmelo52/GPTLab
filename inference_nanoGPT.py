# -*- coding: utf-8 -*-
import torch
from nanoGPT import GPTConfig, nanoGPTModel, nanoGPTTokenizer
import numpy as np
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model_path = 'checkpoints/shakespeare'
#model_path = 'checkpoints/machado_de_assis_conto_CPU'
model_path = 'checkpoints/machado_de_assis_conto'

with open(f'{model_path}/config.json', 'r', encoding='utf-8') as j:
    json_obj = json.loads(j.read())

config = GPTConfig(**json_obj)

tokenizer = nanoGPTTokenizer(model_path) 
model = nanoGPTModel(config)    
model.to(device)   
   
stoi = { ch:i for i,ch in enumerate(tokenizer.vocab) }
penalty_dict = {"u": -0.5, "a": 0.5}  # Penalize "e" and encourage "a"

ckpt_path = f'{model_path}/pytorch_model.bin'     
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
            
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#sentence = 'To be or not to be '
sentence = 'A figura é poética,'
sent = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)

# https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html
output = model.generate(sent, max_new_tokens=1400, temperature=0.8, top_k=None, penalty_factor=None, presence_penalty=penalty_dict, token_to_id=stoi)
print(tokenizer.decode(output[0].tolist()))

# print(model.count_parameters())
# print(model.transformer.h[0].ln_1.a_2)