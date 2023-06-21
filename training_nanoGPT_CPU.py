# -*- coding: utf-8 -*-

import time, datetime
import os
import numpy as np
import torch
from nanoGPT import GPTConfig, nanoGPTModel
      
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# read it in to inspect it
#corpus = 'shakespeare.txt'
corpus = 'machado_de_assis_conto.txt'
model_name = corpus.split(".")[0]
model_path = f'checkpoints/{model_name}_CPU'

with open('corpus/' + corpus, 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# aqui estão todos os caracteres únicos que ocorrem neste texto
chars = sorted(list(set(text)))
vocab_size = len(chars)

config = GPTConfig() 
config.vocab_size = vocab_size
# Ajuste dos hiperparâmetros para uso de uma CPU com 16GB
config.batch_size = 32 # Quantas sequências independentes processaremos em paralelo?
config.max_len = 32 # Qual é o comprimento máximo de contexto (token) para previsões?
config.n_embd = 64
config.n_head = 4
config.n_layer = 4
config.dropout = 0.2
config.max_iters = 5000     

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.max_len, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + config.max_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.max_len]).astype(np.int64)) for i in ix])
    
    if device == 'cuda':
        # pin arrays x,y, que nos permite movê-los para a GPU de forma assíncrona (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(encode("hii there"))
# print(decode(encode("hii there")))

# Train and test splits
data = np.int64(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


model = nanoGPTModel(config)    
model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)  # equivalente a model.forward(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

validationEpoch_loss = []

inicio=time.time()

for iter in range(config.max_iters):    
    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        validationEpoch_loss.append(losses['val'].tolist())

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

fim=time.time()
print("\nTempo total para treinar modelo nanoGPT: %s \n" % (str(datetime.timedelta(seconds=(fim-inicio)))))

model.save_pretrained(model_path, chars)
            
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=1400)[0].tolist()))

from matplotlib import pyplot as plt
plt.plot(validationEpoch_loss, label='val_loss')
plt.legend()
plt.xlabel("eval_interval")
plt.title("val/loss")
plt.show
plt.savefig(f'val_loss_{model_name}_CPU.png', format='png', dpi=300)
