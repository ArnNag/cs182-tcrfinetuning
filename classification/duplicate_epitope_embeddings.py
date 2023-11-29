import numpy as np
import pandas as pd
from tqdm import tqdm

pairs = pd.read_csv('TCREpitopePairs.csv')
epitope_embedding_dict = np.load("embeddings_epitope_no_finetuning.npz") 

epitope_embedding_list = []
for epi in tqdm(pairs['epi']):
    epitope_embedding_list.append(epitope_embedding_dict[epi])

np.save("embeddings_epitope_no_finetuning_duplicated.npy", np.array(epitope_embedding_list))


