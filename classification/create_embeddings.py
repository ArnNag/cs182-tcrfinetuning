import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np

pairs = pd.read_csv('TCREpitopePairs.csv')
epitope_embeddings = []
tcr_embeddings = []


print("loading models")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50") 
nonfinetuned_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

# replace this with fine tuned model
finetuned_model = nonfinetuned_model

print("loaded models")
# get embeddings for epitopes and TCR sequences
for i in range(3):
    print(f"making embeddings for {i}")
    epitope_seq = pairs['epi'][i]
    epitope_input_ids = tokenizer(epitope_seq, return_tensors="pt").input_ids
    epitope_embedding = nonfinetuned_model(input_ids=epitope_input_ids).last_hidden_state.detach().numpy()

    tcr_seq = pairs['tcr'][i]
    tcr_input_ids = tokenizer(tcr_seq, return_tensors="pt").input_ids
    tcr_embedding = finetuned_model(input_ids=tcr_input_ids).last_hidden_state.detach().numpy()
    
    epitope_embeddings.append(epitope_embedding)
    tcr_embeddings.append(tcr_embedding)


# save embeddings
np.savez("embeddings.npz", epi=np.array(epitope_embeddings), tcr=np.array(tcr_embeddings), binding=pairs['binding'])




