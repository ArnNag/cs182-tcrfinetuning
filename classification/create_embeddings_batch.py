import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import torch
import re


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)

def get_embeddings(model, sequences, batch_size):
    sequences = [re.sub(r"[UZOBX]", "", sequence) for sequence in sequences]
    sequences = [" ".join(s) for s in sequences]
    features = []
    currIndex = 0
    while currIndex < len(sequences):
      print(currIndex)
      ids = tokenizer.batch_encode_plus(sequences[currIndex:currIndex+batch_size], add_special_tokens=True, padding=True)
      input_ids = torch.tensor(ids['input_ids']).to(device)
      attention_mask = torch.tensor(ids['attention_mask']).to(device)
      with torch.no_grad():
          embedding = model(input_ids=input_ids,attention_mask=attention_mask)
      embedding = embedding.last_hidden_state.cpu().numpy()
      print(embedding.shape)
      for seq_num in range(len(embedding)):
          seq_len = (attention_mask[seq_num] == 1).sum()
          seq_emd = embedding[seq_num][:seq_len-1]
          seq_emd = np.mean(seq_emd, axis=0)
          features.append(seq_emd)
      currIndex += batch_size

    return features


def load_model(name):
    return T5EncoderModel.from_pretrained(name).to(device).eval()



if __name__ == '__main__':

    print("Reading TCR sequences")
    pairs = pd.read_csv('TCREpitopePairs.csv')
    tcr_seqs = list(pairs["tcr"])
    # model_names = ["Rostlab/prot_t5_xl_uniref50", "checkpoint-100000", "checkpoint-200000"]
    model_names = ["checkpoint-200000"]

    for model_name in model_names:
        print(f"Loading model {model_name}")
        model = load_model(model_name)
        print(f"Making embeddings for model {model_name}")
        tcr_embeddings = get_embeddings(model, tcr_seqs, int(2.7e3))
        print(f"Saving embeddings for model {model_name}")
        model_name_save = model_name.replace("/", "-") 
        np.save(f"tcr_embeddings_{model_name_save}.npy", np.array(tcr_embeddings))

