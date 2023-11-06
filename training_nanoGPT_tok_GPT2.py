# -*- coding: utf-8 -*-

import tiktoken
import numpy as np
import time, datetime
import torch
from nanoGPT import GPTConfig, nanoGPTModel
from generation import generate
      
device = 'cuda' if torch.cuda.is_available() else 'cpu'

corpus = 'machado_de_assis_conto.txt'
model_name = corpus.split(".")[0]
model_path = f'checkpoints/{model_name}_tok_GPT2'

with open('corpus/' + corpus, 'r', encoding='utf-8') as f:
    data = f.read()
    
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
encoding = tiktoken.get_encoding("gpt2")
train_ids = encoding.encode_ordinary(train_data)
val_ids = encoding.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_data = np.array(train_ids, dtype=np.uint16)
val_data = np.array(val_ids, dtype=np.uint16)

# encoding.decode(encoding.encode("Se o leitor é rapaz e dado ao gênio melancólico"))

config = GPTConfig() 
config.vocab_size = encoding.n_vocab

# Ajuste dos hiperparâmetros para uso de uma GPU com 4GB
config.batch_size = 32 # Quantas sequências independentes processaremos em paralelo?
config.max_len = 32 # Qual é o comprimento máximo de contexto (token) para previsões?
config.n_embd = 384
config.n_head = 6
config.n_layer = 6
config.dropout = 0.1
config.max_iters = 5000
     
model = nanoGPTModel(config)    
model.to(device)

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
            model_output = model(X, Y)  # equivalente a model.forward(X, Y)            
            logits, loss = model_output.logits, model_output.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
       
inicio=time.time()

for iter in range(config.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")        

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    model_output = model(xb, yb) 
    loss = model_output.loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

fim=time.time()
print("\nTempo total para treinar modelo nanoGPT: %s \n" % (str(datetime.timedelta(seconds=(fim-inicio)))))

model.save_pretrained(model_path)
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoding.decode(generate(model, context, max_new_tokens=500)[0].tolist()))