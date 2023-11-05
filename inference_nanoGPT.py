# -*- coding: utf-8 -*-
import torch
from nanoGPT import GPTConfig, nanoGPTModel, nanoGPTTokenizer
import numpy as np
import json
import time, datetime

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
# sequências de parada
stop_words =['\n—', '\n\n']
stop_sequences = [tokenizer.encode(stop_word) for stop_word in stop_words]

inicio=time.time()

# https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html
output = model.generate(sent, 
                        temperature=0.8, 
                        max_new_tokens=1400,                         
                        top_k=None, #10,
                        frequency_penalty=None, #0.2,
                        presence_penalty=None, #penalty_dict, 
                        stop_sequence=None, #stop_sequence,
                        token_to_id=stoi
                        )

predicted_tokens = output[0].tolist()
print(tokenizer.decode(predicted_tokens))

fim=time.time()
print("\nTempo total para inderência do modelo nanoGPT: %s \n" % (str(datetime.timedelta(seconds=(fim-inicio)))))


# print(model.count_parameters())
# print(model.transformer.h[0].ln_1.a_2)