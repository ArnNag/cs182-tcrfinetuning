import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from functools import lru_cache

pairs = pd.read_csv('TCREpitopePairs.csv', nrows=10)
epitope_embeddings = []
tcr_embeddings = []


tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50") 
nonfinetuned_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

def get_embedding(seq, model):
    input_ids = tokenizer(seq, return_tensors="pt").input_ids
    embedding = model(input_ids=input_ids).last_hidden_state.detach().numpy()
    return embedding

@lru_cache
def get_tcr_embedding(seq):
    return get_embedding(seq, finetuned_model)

@lru_cache
def get_epitope_embedding(seq):
    return get_embedding(seq, nonfinetuned_model)

# TODO: replace this with fine tuned model
finetuned_model = nonfinetuned_model

# get embeddings for epitopes and TCR sequences
for i in range(10):
    epitope_seq = pairs['epi'][i]
    epitope_embedding = get_epitope_embedding(epitope_seq)

    tcr_seq = pairs['tcr'][i]
    tcr_embedding = get_tcr_embedding(tcr_seq)
    
    epitope_embeddings.append(epitope_embedding)
    tcr_embeddings.append(tcr_embedding)


# save embeddings
np.savez(epi=np.array(epitope_embeddings), tcr=np.array(tcr_embeddings), binding=pairs['binding'])

