
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
from json import dumps
from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass
class GPTConfig:
    # hiperparâmetros
    # valores de configuração padrão projetados para treinar um nanoGPT
    vocab_size: int = 50257  # Tamanho default para o GPT2
    batch_size: int = 32 # Quantas sequências independentes processaremos em paralelo?
    max_len: int = 64 # Qual é o comprimento máximo de contexto (token) para previsões?
    max_iters: int = 5000
    eval_interval: int = 100
    learning_rate: float = 3e-4
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout:float = 0.1
    bias: bool = False # True: viés em Linears e LayerNorms, como GPT-2. False: um pouco melhor e mais rápido
    # ------------
    
    @property
    def __dict__(self):
        return asdict(self)

    @property
    def json(self):
        return dumps(self.__dict__,indent=4,ensure_ascii=False) 


def new_gelu(x):
    """
     Implementação da função de ativação GELU atualmente no repositório Google BERT (idêntico ao OpenAI GPT).
     Referência: papel Gaussian Error Linear Units (GELU): https://arxiv.org/abs/1606.08415
    """
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    "Constroi um módulo layernorm."

    def __init__(self, ndim, bias, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ndim))
        self.b_2 = nn.Parameter(torch.zeros(ndim)) 
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 
       


def attention(q, k, v, mask_att, attn_dropout, mask=None, dropout=None):   
    # Suspeitamos que para grandes valores de n_embd, os produtos escalares crescem em magnitude, 
    # empurrando a função softmax para regiões onde ela produz gradientes extremamente pequenos
    # Para neutralizar esse efeito, escalamos os produtos escalares por 1/raiz(n_embd//n_head (1.0 / math.sqrt(k.size(-1)))
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if mask is not None:
        att = att.masked_fill(mask_att == 0, float('-inf'))

    att = F.softmax(att, dim=-1)
    att_wei = att.clone()

    if dropout != 0.0:
        att = attn_dropout(att)
    z = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return z, att_wei
        

class MultiHeadedAttention(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.d_k = config.n_embd // config.n_head
        # key, query, value projections for all n_head, but in a batch        
        self.q_linear = nn.Linear(config.n_embd, config.n_embd)
        self.v_linear = nn.Linear(config.n_embd, config.n_embd)
        self.k_linear = nn.Linear(config.n_embd, config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        # Flash Attention: verificar se versão do PyTorch tem suporte
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            if self.layer_idx == 0:
                print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # máscara causal para garantir que a atenção seja aplicada apenas à esquerda na sequência de entrada
            # tril é uma matriz triangular inferior. não é um parâmetro
            # do modelo, então o atribuímos ao módulo usando register_buffer
            self.register_buffer("masks", torch.tril(torch.ones(config.max_len, config.max_len))
                                        .view(1, 1, config.max_len, config.max_len))

    def forward(self, x, mask=None):
        B, T, C = x.size() # B:tamanho do batch, T:comprimento da sequência, C:dimensionalidade do embedding (n_embd)

        # executar operação linear e dividir em N n_head
        k = self.k_linear(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2) # (B, nh, T, hs)
        q = self.q_linear(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2) # (B, nh, T, hs)
        v = self.v_linear(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2) # (B, nh, T, hs)        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # atenção eficiente usando Flash Attention CUDA kernels
            z = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout, is_causal=True)
        else:
            # implementação manual do mecanismo de atenção
            z, att_wei  = attention(q, k, v, self.masks[:,:,:T,:T], self.attn_dropout, mask, self.dropout)
            
        z = z.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        attn = self.resid_dropout(self.c_proj(z))
        return attn, att_wei 
    
    
# MultiLayer perceptron    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadedAttention(config, layer_idx=layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        hidden_states = self.ln_1(x)
        attn_outputs, att_wei  = self.attn(hidden_states, mask=True)
        # residual connection
        hidden_states = x + attn_outputs
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states, attn_outputs, att_wei         



class nanoGPTModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        # with open(f'{model_path}/config.json', 'r', encoding='utf-8') as j:
        #     json_obj = json.loads(j.read())
        
        #self.config = GPTConfig(**json_obj)
        self.config = config

        assert self.config.vocab_size is not None
        assert self.config.max_len is not None
        assert self.config.max_len <= 256  # comprimento máximo do contexto que pode ser configurado

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.max_len, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(self.config, layer_idx=i) for i in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)        
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
         Retorna o número de parâmetros no modelo.
         Para contagem de não embedding (padrão), as embeddings de posição são subtraídas.
         As embeddings de token também, exceto devido ao compartilhamento de parâmetros
         na verdade, os parâmetros são usados como pesos na camada final, então nós os incluímos.
         """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_len, f"Não é possível encaminhar a sequência de comprimento {t}, o tamanho do bloco é apenas {self.config.max_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings na forma (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings na forma (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        all_self_attentions = []
        all_hidden_states = []
        all_att_weights = []
        for block in self.transformer.h:
            x, att, att_wei  = block(x)
            all_self_attentions.append(att)
            all_hidden_states.append(x)
            all_att_weights.append(att_wei)
            
        last_hidden_state = self.transformer.ln_f(x)
        all_hidden_states = [self.transformer.ln_f(hidden_state) for hidden_state in all_hidden_states]

        if targets is not None:
            # se recebermos alguns alvos desejados, calcule também a perda
            logits = self.lm_head(last_hidden_state)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # mini-otimização de tempo de inferência: encaminhar apenas o lm_head na última posição
            logits = self.lm_head(last_hidden_state[:, [-1], :]) # nota: usando a lista [-1] para preservar a dimensão de tempo(T)
            loss = None

        #return logits, loss, att_wei
        return CausalLMOutput(
                last_hidden_state=last_hidden_state,
                logits=logits,
                loss=loss,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,                
                attentions_weights=all_att_weights,   
            )
    

    def save_pretrained(self, save_directory, vocab=None):
        if os.path.isfile(save_directory):
            print(f"O caminho fornecido ({save_directory}) deve ser um diretório, não um arquivo")
            return
        
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"salvando checkpoint para {save_directory}")
        ckpt_path = os.path.join(save_directory, 'pytorch_model.bin')
        
        try:
            torch.save(self.state_dict(), ckpt_path)
            print("Salvou com sucesso o modelo para{}".format(ckpt_path))
        except Exception as e:
            print(f"Erro ao salvar o modelo no checkpoint. {e}")
        
        with open(f"{save_directory}/config.json", "w", encoding='utf-8') as jsonFile:
            jsonFile.write(self.config.json) 
        
        if vocab is not None:
            with open(f"{save_directory}/vocab.txt", "w", encoding='utf-8') as f:
                f.write(''.join(vocab)) 
            
    
    def count_parameters(self):
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                print(f"{name}: {num_params}")
                total_params += num_params
        print(f"Total Trainable Parameters: {total_params}")
        
    
    def apply_frequency_penalty(self, logits, generated_tokens, frequency_penalty):
        """
        Apply frequency penalty to the logits.

        Args:
        - logits (np.array): The raw output logits from the language model for the next token prediction.
        - generated_tokens (list of int): List of token IDs that have already been generated.
        - penalty_factor (float): The factor by which the logits of already generated tokens will be penalized.

        Returns:
        - np.array: Modified logits.
        """

        logits = logits.squeeze(0).squeeze(0).cpu().numpy()
        generated_tokens = generated_tokens.squeeze(0).cpu()
        # Count the frequency of each token in the generated tokens
        token_counts = np.bincount(generated_tokens, minlength=len(logits))
        
        # Apply penalty to logits of tokens that have already been generated
        penalized_logits = logits - frequency_penalty * token_counts

        tensor_penalized_logits = torch.tensor(penalized_logits).unsqueeze(0).unsqueeze(0)
        return tensor_penalized_logits.to('cuda')
    
    
    def apply_presence_penalty(self, logits, token_to_id, penalty_dict):
        """
        Apply presence penalty to the logits.

        Args:
        - logits (np.array): The raw output logits from the language model for the next token prediction.
        - token_to_id (dict): Dictionary that maps tokens (words or subwords) to their respective IDs.
        - penalty_dict (dict): Dictionary that specifies the penalty values for certain tokens.

        Returns:
        - np.array: Modified logits.
        """
        logits = logits.squeeze(0).squeeze(0).cpu().numpy()

        for token, penalty in penalty_dict.items():
            if token in token_to_id:
                token_id = token_to_id[token]
                logits[token_id] += penalty

        tensor_penalized_logits = torch.tensor(logits).unsqueeze(0).unsqueeze(0)
        return tensor_penalized_logits.to('cuda')


    @torch.no_grad()
    def generate(self, idx, temperature=1.0, max_new_tokens=500, top_k=None, frequency_penalty=None, presence_penalty=None, token_to_id=None, stop_sequence=None):
        """
        Pegue uma sequência de condicionamento de índices idx (LongTensor de forma (b,t)) e complete
        a sequência max_new_tokens vezes, alimentando as previsões de volta ao modelo a cada vez.
        Muito provavelmente você vai querer certificar-se de estar no modo de operação model.eval() para isso.
        """
        b, t = idx.size()
        assert t <= self.config.max_len, f"Não é possível encaminhar a sequência de comprimento {t}, o tamanho máximo de tokens na sentença é {self.config.max_len}"                

        if stop_sequence is not None:
            stop_sequence_id = token_to_id[stop_sequence]

        for _ in range(max_new_tokens):
            # se o contexto da sequência estiver crescendo muito, devemos cortá-lo em max_len
            if idx.size(1) <= self.config.max_len:
                idx_cond = idx  
            else: 
                idx_cond = idx[:, -self.config.max_len:]

            # encaminhar o modelo para obter os logits para o índice na sequência
            model_output = self(idx_cond)
            logits = model_output.logits
            
            # A penalidade de frequência é utilizada para reduzir a repetição de palavras ou frases no texto gerado.
            # que já foram produzidas na sessão de geração atual. 
            if frequency_penalty is not None:
                logits = self.apply_frequency_penalty(logits, idx_cond, frequency_penalty)

            # a penalidade de presença foca em ajustar a probabilidade de determinados tokens ou frases serem gerados, 
            # seja para aumentá-la ou diminuí-la.
            if presence_penalty is not None:
                logits = self.apply_presence_penalty(logits, token_to_id, presence_penalty)

            # pegue os logits na etapa final e dimensione pela temperatura desejada
            temperature = 0.0001 if temperature == 0.0 else temperature
            logits = logits[:, -1, :] / temperature
            
            # top_k: top probabilidades, opcionalmente, corte os logits para apenas as k principais opções
            if top_k is not None:
                # Retorna os k maiores elementos do tensor de entrada fornecido ao longo de uma determinada dimensão.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')                
            
            # aplique softmax para converter logits em probabilidades (normalizadas)
            probs = F.softmax(logits, dim=-1)
            # tire uma amostra da distribuição 
            idx_next = torch.multinomial(probs, num_samples=1)
            #idx_next = torch.argmax(probs, dim=1).unsqueeze(1)  
                      
            # anexe o índice amostrado à sequência em execução e continue
            idx = torch.cat((idx, idx_next), dim=1)

            # As sequências de parada, também conhecidas como sequências de terminação ou marcadores de final de texto, 
            # são sequências predefinidas de tokens que sinalizam a um modelo de linguagem para parar de gerar mais texto.
            if stop_sequence is not None and stop_sequence_id in idx:
                break

        return idx
      




from collections import OrderedDict, UserDict
from typing import Optional, Tuple

# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py
@dataclass
class CausalLMOutput(OrderedDict):
    """
    Classe base para saídas de modelo de linguagem causal (ou autorregressivo).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequência de estados ocultos na saída da última camada do modelo.
            (Representação contextual do token)
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Predição dos scores da camada Linear Head (pontuações para cada token de vocabulário antes do SoftMax).
        loss (`torch.FloatTensor` de forma `(1,)`, *opcional*, retornado quando `labels` é fornecido):
            Perda de modelagem de linguagem (para previsão do próximo token).
        hidden_states (`tuple(torch.FloatTensor)`):
            Tupla de `torch.FloatTensor` (um para a saída dos embeddings, se o modelo tiver uma camada de embedding, +
             um para a saída de cada camada) de forma `(batch_size, sequence_length, hidden_size)`.
             Estados ocultos do modelo na saída de cada camada mais as saídas de embeddings iniciais opcionais.
        attentions (`tuple(torch.FloatTensor)`):
            Tupla de `torch.FloatTensor` (um para cada camada) de forma `(batch_size, num_heads, sequence_length,
             comprimento_sequência)`.
        attentions_weights (`tuple(torch.FloatTensor)`):
            após a atenção softmax, usado para calcular a média ponderada nas cabeças de auto-atenção.
    """

    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None
    hidden_states: Tuple[torch.FloatTensor] = None
    attentions: Tuple[torch.FloatTensor] = None    
    attentions_weights: Tuple[torch.FloatTensor] = None 



class nanoGPTTokenizer():

    def __init__(self, model_path):
        with open(f'{model_path}/vocab.txt', 'r', encoding='utf-8') as f:
            self.vocab = list(f.read())
            # create a mapping from characters to integers
            self.stoi = { ch:i for i,ch in enumerate(self.vocab) }
            self.itos = { i:ch for i,ch in enumerate(self.vocab) }

    def encode(self, text):        
        encode_text = [self.stoi[c] for c in text] # encoder: take a string, output a list of integers
        return encode_text

    def decode(self, list):        
        decode_text = ''.join([self.itos[i] for i in list]) # decoder: take a list of integers, output a string
        return decode_text
    
    @property
    def vocab_size(self):
        return len(self.vocab)




