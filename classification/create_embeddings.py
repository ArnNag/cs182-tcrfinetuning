import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import torch

pairs = pd.read_csv('TCREpitopePairs.csv', nrows=3)
pairs['epi'] = pairs.apply(lambda row : " ".join(row["epi"]), axis = 1)
pairs['tcr'] = pairs.apply(lambda row : " ".join(row["tcr"]), axis = 1)
epitope_embeddings = []
tcr_embeddings = []


print("loading models")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50") 
nonfinetuned_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

def get_embedding(seq, model):
    input_ids = tokenizer(seq, return_tensors="pt").input_ids
    hidden_states = model.encoder(input_ids=input_ids).last_hidden_state.detach()
    hidden_states_pooled = torch.mean(hidden_states, dim=1)
    return hidden_states_pooled

@lru_cache(maxsize=4096)
def get_tcr_embedding(seq):
    return get_embedding(seq, finetuned_model)

@lru_cache(maxsize=4096)
def get_epitope_embedding(seq):
    return get_embedding(seq, nonfinetuned_model)

# TODO: replace this with fine tuned model
finetuned_model = nonfinetuned_model

print("loaded models")
# get embeddings for epitopes and TCR sequences
for i in tqdm(range(3)):
    epitope_seq = pairs['epi'][i]
    epitope_embedding = get_epitope_embedding(epitope_seq)

    tcr_seq = pairs['tcr'][i]
    tcr_embedding = get_tcr_embedding(tcr_seq)
    
    epitope_embeddings.append(epitope_embedding)
    tcr_embeddings.append(tcr_embedding)


# save embeddings
np.savez("embeddings_no_finetuning.npz", epi=np.array(epitope_embeddings), tcr=np.array(tcr_embeddings), binding=pairs['binding'])

