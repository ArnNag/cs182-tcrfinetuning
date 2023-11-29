# Fine-tuning Protein Large Language Models to Develop Comprehensive T-cell Receptor Embeddings
## Downloading the Fine-Tuned ProtT5 model
The latest fine-tuned version of the ProtT5 model to T cell receptor sequences can be downloaded via:
```
gsutil -m cp -r "gs://tcrcheckpoints/checkpoint-50000" .
```
Once downloaded, to load the model into your code run:
```
import torch
from transformers import T5EncoderModel, T5Tokenizer
import gc

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("checkpoint-50000/")
```
These commands are done for you in some of the figure-reproducing scripts

## UMAP Figure Reproduction (Figure 2) and Generating Embeddings
The first section of the ProtT5_Embedding notebook (titled Output Fine-tuned Embeddings) generates the TCR embeddings from their amino acid sequences. The second section (titled Plot Fine-tuned Embeddings) contains the code necessary to reproduce the UMAP plots (Figure 2). Since the full dataset is very large, we provide a subset of the dataset

## Hierarchical Clustering Reproduction (Figure 3; left)
To reproduce the left plot in Figure 3 (since the right plot is from the catELMo paper), run the clustering.ipynb notebook. To download the embeddings used for this analysis, use the following dropbox link (in the form of a pickled list): https://www.dropbox.com/scl/fi/7qef6cb06fhozfvdzeuth/McPAS_Embeddings_FineTuned.pkl?rlkey=42r6knr7k4skw5ee0lz1l6eku&dl=0

## Fine-Tuning Procedure
Our LoRA fine-tuning procedure is in prott5/Fine-Tuning/PT5_LoRA_TCR_pilot.ipynb. Our linear probing procedure is outlined in ProtT5_Embedding.ipynb. Although pre-training these yourself may take hours, these notebooks contain all the relevant information needed to reproduce Figure 1. 

