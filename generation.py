import torch
from torch.nn import functional as F
import numpy as np

#max_input_tokens = 512

def apply_frequency_penalty(logits, generated_tokens, frequency_penalty):
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
    return tensor_penalized_logits



def apply_presence_penalty(logits, token_to_id, penalty_dict):
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
    return tensor_penalized_logits



def detect_stop_sequence(idx, stop_sequences):
    """
    As sequências de parada, também conhecidas como sequências de terminação ou marcadores de final de texto, 
    são sequências predefinidas de tokens que sinalizam a um modelo de linguagem para parar de gerar mais texto.
    """
    for stop_sequence in stop_sequences:
        s = len(stop_sequence)
        A = torch.tensor(stop_sequence)
        B = idx[0,-s:].cpu()

        if all(A == B):
            return True

    return False              



@torch.no_grad()
def generate(model,
             idx, 
             temperature=0.8, 
             max_new_tokens=500, 
             top_k=None, 
             frequency_penalty=None, 
             presence_penalty=None, 
             token_to_id=None, 
             stop_sequence=None):
    """
    Pegue uma sequência de condicionamento de índices idx (LongTensor de forma (b,t)) e complete
    a sequência max_new_tokens vezes, alimentando as previsões de volta ao modelo a cada vez.
    Muito provavelmente você vai querer certificar-se de estar no modo de operação model.eval() para isso.
    """
    device = idx.device
    b, t = idx.size()
    assert t <= model.config.max_len, f"Não é possível encaminhar a sequência de comprimento {t}, o tamanho máximo de tokens na sentença é {model.config.max_len}"                

    for _ in range(max_new_tokens):
        # se o contexto da sequência estiver crescendo muito, devemos cortá-lo em max_len
        if idx.size(1) <= model.config.max_len:
            idx_cond = idx  
        else: 
            idx_cond = idx[:, -model.config.max_len:]

        # realize uma inferência (logits) no modelo nanoGPT com base em uma sequência de entrada (idx_cond).
        model_output = model(idx_cond)
        logits = model_output.logits
        
        # A penalidade de frequência é utilizada para reduzir a repetição de palavras ou frases no texto gerado.
        # que já foram produzidas na sessão de geração atual. 
        if frequency_penalty is not None:
            logits = apply_frequency_penalty(logits, idx_cond, frequency_penalty).to(device)

        # a penalidade de presença foca em ajustar a probabilidade de determinados tokens ou frases serem gerados, 
        # seja para aumentá-la ou diminuí-la.
        if presence_penalty is not None:
            logits = apply_presence_penalty(logits, token_to_id, presence_penalty).to(device)

        # pegue os logits na etapa final e dimensione pela temperatura desejada
        temperature = 0.0001 if temperature == 0.0 else temperature
        logits = logits[:, -1, :] / temperature
        
        # top_k: top probabilidades, opcionalmente, corte os logits para apenas as k principais opções
        if top_k is not None:
            # torch.topk - retorna os k maiores elementos do tensor de entrada fornecido ao longo de uma determinada dimensão.
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')                
        
        # aplique softmax para converter logits em probabilidades (normalizadas)
        probs = F.softmax(logits, dim=-1)
        # tire uma amostra da distribuição 
        idx_next = torch.multinomial(probs, num_samples=1)
        #idx_next = torch.argmax(probs, dim=1).unsqueeze(1)  
                  
        # anexe o índice amostrado à sequência em execução e continue
        idx = torch.cat((idx, idx_next), dim=1)
        
        if stop_sequence is not None and detect_stop_sequence(idx, stop_sequence):
            break

    return idx