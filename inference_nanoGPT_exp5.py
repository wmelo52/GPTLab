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
            


# https://github.com/cthiriet/gpt-lab/blob/main/babygpt-trip.ipynb

import matplotlib.pyplot as plt
from torch.nn import functional as F

def plot_probs(probs, stoi=tokenizer.stoi):
    """ plot the probabilities histogram of the next token """
    # plot as histogram and display tokens with highest probability    
    plt.figure(figsize=(12, 4), dpi=200)
    plt.bar(np.arange(len(probs[0])), probs[0])  
    # stoi['\\n'] = stoi['\n']
    # del stoi['\n']
    # stoi['\\s'] = stoi[' ']
    # del stoi[' ']   
    plt.xticks(np.arange(len(probs[0])), stoi)
    plt.xticks(rotation=0, fontsize=7)

    # display the exact proba values
    for i, p in enumerate(probs[0]):
        if p > 0.01:
            plt.text(i, p, f"{p:.3f}", va='bottom', ha='center')
      
    plt.show()        



# plot the output distribution of the trained model
context = torch.randint(config.vocab_size, (1, config.max_len), dtype=torch.long, device=device)
print(tokenizer.decode(context[0].cpu().numpy()))
print(context.shape)
logits, _, _ = model(context) # (B, T, vocab_size)
print(logits.shape)

logits = logits[:, -1, :] # becomes (B, C)
# apply softmax to get probabilities
probs = F.softmax(logits, dim=-1) # (B, C)
probs = probs.detach().cpu().numpy()

plot_probs(probs)




# plot the attention weights
def plot_attentions_wei(sentence, device, config, tokenizer, model):    
    data = np.int64(tokenizer.encode(sentence))
    sent = torch.from_numpy(data).unsqueeze(0).to(device)

    logits, loss, attention_weights = model(sent)

    labels = sentence
    target = sentence

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
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(target);
        ax.set_title(f"head {i+1}")

sentence = 'A figura é poética'
plot_attentions_wei(sentence, device, config, tokenizer, model)
    
    