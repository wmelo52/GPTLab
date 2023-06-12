# -*- coding: utf-8 -*-
import torch
import json
from nanoGPT import GPTConfig, nanoGPTModel, nanoGPTTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model_path = 'checkpoints/shakespeare'
model_path = 'checkpoints/machado_de_assis_conto_CPU'
#model_path = 'checkpoints/machado_de_assis_conto'

model_name = model_path.split("/")[1]

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


# ---------------------- Making Word cluster plot positional embedding---------------------------- 
pos = torch.arange(0, config.max_len, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
pos_emb = model.transformer.wpe(pos) # position embeddings na forma (1, t, n_embd)
pos_emb = pos_emb.squeeze(0)

with open("data/pos_emb_384.txt", "w", encoding='utf-8') as f:
    f.write(str(config.max_len) + ' ' + str(config.n_embd) + '\n') 
    for i,vector in enumerate(pos_emb):
        f.write(str(i) + ' ' + ' '.join(['{:.6f}'.format(x_) for x_ in vector]) + '\n') 
    
 
# Visualizing word embeddings
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

wv_name = 'pos_emb_384.txt'

def visualize_embeddings(model_path, wv_name, config, group1, group2):
    # Loading embedding model
    model_wv = KeyedVectors.load_word2vec_format('data/' + wv_name, binary=False, unicode_errors="ignore")

    tsne = TSNE(n_components=2, random_state=0)
    all_word_vectors_matrix = model_wv.vectors
    T = tsne.fit_transform(all_word_vectors_matrix)

    points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, T[model_wv.key_to_index[word]])            
            for word in model_wv.key_to_index
        ]
    ],
    columns=["word", "x", "y"]
    )
    labels = points["word"]   
 
    #sns.set_context("poster")
    plt.rcParams["figure.autolayout"] = True   

    colors = ["black", "red", "b"]  
    colormap = []
    for w in points["word"]:
        if w in group1:
            colormap.append(colors[1])
        elif w in group2:
            colormap.append(colors[2])
        else:
            colormap.append(colors[0])

    ax = points.plot.scatter("x", "y", s=150, c=colormap, figsize=(19, 10))  # c='g'
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x+0.20, y+0.20), xytext=(0, 0), textcoords='offset points', fontsize=16)
    
    fig = ax.get_figure()
    fig.savefig(os.path.join("images", f"{model_name}_{wv_name[:7]}_{config.max_iters}.png"))


vowels_group = ["a","e","i","o","u"]
visualize_embeddings(model_path, wv_name, config, vowels_group, vowels_group)




# ------------------------- Making Word cluster plot - token embedding-------------------------------
pos = torch.arange(0, config.vocab_size, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
tok_emb = model.transformer.wte(pos)
tok_emb = tok_emb.squeeze(0)

tokenizer.vocab[0] = '\\n'
tokenizer.vocab[1] = '\\s'
with open("data/tok_emb_384.txt", "w", encoding='utf-8') as f:
    f.write(str(config.vocab_size) + ' ' + str(config.n_embd) + '\n') 
    for i,vector in enumerate(tok_emb):
        f.write(tokenizer.vocab[i] + ' ' + ' '.join(['{:.6f}'.format(x_) for x_ in vector]) + '\n') 
        
wv_name = 'tok_emb_384.txt'
vowels_group = ["a","e","i","o","u"]
numbers_group = ["0","1","2","3","4","5","6","7","8","9"]
visualize_embeddings(model_path, wv_name, config, vowels_group, numbers_group)
        