import pandas as pd

from src.data.data_processing import *

immrep22_folder = 'data/training_data/'
epitopes = list_epitopes(immrep22_folder)
immrep22 = load_complete_data(epitopes, immrep22_folder, ['TRA_CDR3', 'TRB_CDR3'])
immrep22 = immrep22[immrep22['Label'] == 1]

immrep23 = pd.read_csv('data/VDJdb_paired_chain.csv')
immrep23 = immrep23[['Peptide', 'CDR3a_extended', 'CDR3b_extended']]
immrep23.rename(columns={'Peptide': 'Epitope', 'CDR3a_extended': 'TRA_CDR3', 'CDR3b_extended': 'TRB_CDR3'}, inplace=True)
immrep23['Label'] = int(1)

common_rows = pd.merge(immrep22, immrep23, how='inner', on=['Epitope', 'TRA_CDR3', 'TRB_CDR3', 'Label'])
immrep23 = immrep23[~immrep23.isin(common_rows)].dropna().reset_index(drop=True)

common_epitopes = immrep22[['Epitope']].merge(immrep23[['Epitope']], on='Epitope')
rows_to_move = immrep23[immrep23['Epitope'].isin(common_epitopes['Epitope'])]
immrep22 = pd.concat([immrep22, rows_to_move], ignore_index=True)
immrep23 = immrep23[~immrep23['Epitope'].isin(common_epitopes['Epitope'])].reset_index(drop=True)

df = pd.concat([immrep22, immrep23], ignore_index=True)

epitopes = df['Epitope'].unique().tolist()

for epitope in epitopes:
    count = (df['Epitope'] == epitope).sum()
    filtered_df = df[df['Epitope'] != epitope]
    negatives = filtered_df.sample(n=count*5)
    negatives['Epitope'] = epitope
    negatives['Label'] = int(0)
    df = pd.concat([df, negatives], ignore_index=True)

df.to_csv('data/new.csv', sep=' ')
