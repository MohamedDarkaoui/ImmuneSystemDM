import pandas as pd

from src.data.data_processing import *

immrep22_folder = 'data/training_data/'
epitopes = list_epitopes(immrep22_folder)
immrep22 = load_complete_data2(epitopes, immrep22_folder, ['TRA_CDR3', 'TRB_CDR3'])
immrep22 = immrep22[immrep22['Label'] == 1]

immrep23 = pd.read_csv('data/VDJdb_paired_chain.csv')
immrep23 = immrep23[['Peptide', 'CDR3a_extended', 'CDR3b_extended']]
immrep23.rename(columns={'Peptide': 'Epitope', 'CDR3a_extended': 'TRA_CDR3', 'CDR3b_extended': 'TRB_CDR3'}, inplace=True)
immrep23['Label'] = int(1)
immrep23 = immrep23.dropna()

df = pd.concat([immrep22, immrep23], ignore_index=True)
df = df.drop_duplicates().reset_index(drop=True)

epitopes = df['Epitope'].unique().tolist()

for epitope in epitopes:
    count = (df['Epitope'] == epitope).sum()
    filtered_df = df[df['Epitope'] != epitope]
    negatives = filtered_df.sample(n=count*5)
    negatives['Epitope'] = epitope
    negatives['Label'] = int(0)
    df = pd.concat([df, negatives], ignore_index=True)

df['Label'] = df['Label'].astype(int)

test_folder = 'data/true_set/'
test_epitopes = list_epitopes(test_folder)
df_test = load_complete_data2(test_epitopes, test_folder, ['TRA_CDR3', 'TRB_CDR3'])

merged_df = pd.merge(df, df_test, on=list(df.columns), how='outer', indicator=True)
unique_to_df = merged_df[merged_df['_merge'] == 'left_only']
unique_to_df = unique_to_df.drop(columns=['_merge'])
unique_to_df = unique_to_df.sort_values(by='Epitope')

nan_df = unique_to_df.isna()
rows_with_nan = nan_df.any(axis=1)
nan_rows = unique_to_df[rows_with_nan]
unique_to_df.to_csv('data/new_train.csv', sep=' ', index=False)
# print(df)

