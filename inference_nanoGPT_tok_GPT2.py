# -*- coding: utf-8 -*-
import torch
from nanoGPT import GPTConfig, nanoGPTModel
import tiktoken
import numpy as np
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'checkpoints/machado_de_assis_conto_tok_GPT2'

with open(f'{model_path}/config.json', 'r', encoding='utf-8') as j:
    json_obj = json.loads(j.read())

config = GPTConfig(**json_obj)

# encode with tiktoken gpt2 bpe
encoding = tiktoken.get_encoding("gpt2")
model = nanoGPTModel(config)    
model.to(device)   
   
ckpt_path = f'{model_path}/pytorch_model.bin'     
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
            
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#sentence = 'To be or not to be '
sentence = 'A figura é poética, mas não é a da heroína do romance.'
data = np.int64(encoding.encode(sentence))
sent = torch.from_numpy(data).unsqueeze(0).to(device)
print(encoding.decode(model.generate(sent, max_new_tokens=500)[0].tolist()))

#print(model.count_parameters())

#[encoding.decode_single_token_bytes(token) for token in encoding.encode('A figura é poética!')]