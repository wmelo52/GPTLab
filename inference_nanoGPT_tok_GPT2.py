# -*- coding: utf-8 -*-
import torch
from nanoGPT import GPTConfig, nanoGPTModel
import tiktoken
import numpy as np
import json
import time, datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = 'checkpoints/machado_de_assis_conto_tok_GPT2'

with open(f'{model_path}/config.json', 'r', encoding='utf-8') as j:
    json_obj = json.loads(j.read())

config = GPTConfig(**json_obj)

# encode with tiktoken gpt2 bpe
tokenizer = tiktoken.get_encoding("gpt2")
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
sent = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(device)
# sequências de parada
stop_words =['\n-', '?']
stop_sequences = [tokenizer.encode(stop_word) for stop_word in stop_words]

inicio=time.time()

# top_k: top probabilidades
output = model.generate(sent, 
                        temperature=0.8,
                        max_new_tokens=500,                          
                        top_k=None, 
                        frequency_penalty=None,
                        stop_sequence=None #stop_sequences
                       )

print(tokenizer.decode(output[0].tolist()))

fim=time.time()
print("\nTempo total para inderência do modelo nanoGPT: %s \n" % (str(datetime.timedelta(seconds=(fim-inicio)))))

#print(model.count_parameters())
#[encoding.decode_single_token_bytes(token) for token in encoding.encode('A figura é poética!')]



import matplotlib.pyplot as plt

# plot the attention weights
def plot_attentions_wei(sentence, device, config, encoding, model):        
    sent = torch.tensor(encoding.encode(sentence)).unsqueeze(0).to(device)

    model_output = model(sent) 
    attention_weights = model_output.attentions_weights[-1]

    a = [encoding.decode_single_token_bytes(token) for token in encoding.encode(sentence)]
    decoded = [t.decode() for t in a]

    labels = decoded
    target = decoded

# visualize all attention heads on a single plot
    fig, axs = plt.subplots(1, config.n_head, figsize=(20, 5), dpi=200)

    for i in range(config.n_head):
        att_wei_i = attention_weights.squeeze(0)[i].detach().cpu().numpy()
        ax = axs[i]
        ax.matshow(att_wei_i)

    # display the number on each cell
        for (k, j), z in np.ndenumerate(att_wei_i):
            ax.text(j, k, '{:0.2f}'.format(z), ha='center', va='center', color='white', fontsize=3)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(target)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticklabels(target, fontsize=8);
        ax.set_title(f"head {i+1}")

sentence = 'A figura é poética, mas não é a da heroína do romance.'
#sentence = 'Espero que o romance da nossa amizade não termine no primeiro capítulo.'
#plot_attentions_wei(sentence, device, config, encoding, model)
    