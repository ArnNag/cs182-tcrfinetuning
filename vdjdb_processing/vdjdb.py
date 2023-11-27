import pandas as pd

vdj_df = pd.read_csv('SearchTable-2023-11-20 22 48 28.999.tsv', sep='\t', header=0, index_col=False, usecols=[0, 1, 2, 3, 4, 9])

epitope_counts = vdj_df['Epitope'].value_counts()

print("number of epitopes:")
print(len(epitope_counts), "\n")

print("epitopes with less than 25 samples:") # samples are paired
print(sum(epitope_counts < 50), "\n")

v_j_epitope_groups = vdj_df.groupby(by=['V','J', 'Epitope']).size()

print("number of unique V-J-Epitope combinations:")
print(len(v_j_epitope_groups), "\n")

print("V-J-Epitope combinations with more than 1 sample:")
print(sum(v_j_epitope_groups > 2), "\n")

