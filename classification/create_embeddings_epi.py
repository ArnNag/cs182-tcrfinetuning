import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from create_embeddings_batch import get_embeddings 

pairs = pd.read_csv('TCREpitopePairs.csv')
unique_epitopes = pairs['epi'].unique()
print(f"Number of unique epitopes: {len(unique_epitopes)}")

print("creating embeddings")
epitope_embeddings_list = get_embeddings(unique_epitopes, int(2.5e3))

epitope_embeddings = dict()
for epitope, epitope_embedding in zip(unique_epitopes, epitope_embeddings_list):
    epitope_embeddings[epitope] = epitope_embedding


# save embeddings
np.savez("embeddings_epitope_no_finetuning.npz", **epitope_embeddings)

