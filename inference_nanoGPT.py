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
   
ckpt_path = f'{model_path}/pytorch_model.bin'     
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
            
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#sentence = 'To be or not to be '
sentence = 'A figura é poética, mas não é a da heroína do romance.'
data = np.int64(tokenizer.encode(sentence))
sent = torch.from_numpy(data).unsqueeze(0).to(device)
print(tokenizer.decode(model.generate(sent, max_new_tokens=1400)[0].tolist()))