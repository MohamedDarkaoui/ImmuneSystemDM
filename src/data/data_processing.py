import os

import pandas
import pandas as pd
import numpy as np
from src.external.bio.peptide_feature import parse_features, parse_operator
from src.external.bio.feature_builder import CombinedPeptideFeatureBuilder


def list_epitopes(folder_name: str):
    """
    Given a folder name, it returns all file names without the .txt extension, except a few exceptions
    """
    files = os.listdir(folder_name)
    epitopes = [
        file.replace('.txt', '') for file in files if file != 'README.txt' and not file.startswith('test')
    ]
    return epitopes


def concat_columns(df: pandas.DataFrame, tcr_chains: list):
    """
    Concatenates all strings in tcr_chain list, creates a new column in df and removes all columns in df
    that are in tcr_chains.
    """
    col_name = ''.join(tcr_chains)
    df[col_name] = df[tcr_chains].apply(lambda x: ''.join(x), axis=1)
    df.drop(columns=tcr_chains, inplace=True)
    return df


def load_epitope_tcr_data(folder_name: str,  epitope_name: str, tcr_chains: list):
    """
    Creates a data frame that has three columns: [Epitope, tcr_chain[0]...tcr_chain[-1], Label]
    for one given epitope

    @param folder_name: folder where files named after one epitope are found, they contain labeled tcr chains
    @param epitope_name: the given epitope
    @param tcr_chains: the different chains we want in our data frame
    @return: a data frame
    """
    df = pd.read_csv(folder_name + epitope_name + '.txt', sep='\t')
    columns = ['Epitope'] + tcr_chains + ['Label']
    df['Epitope'] = epitope_name
    df = df[columns]
    df['Label'] = df['Label'].replace(-1, 0)
    if len(tcr_chains) > 1:
        df = concat_columns(df, tcr_chains)
    return df


def load_complete_data(epitopes: list, folder_name: str, tcr_chains: list):
    """
    Creates a data frame that has three columns: [Epitope, tcr_chain[0]...tcr_chain[-1], Label]
    for oll epitopes

    @param epitopes: the given epitopes
    @param folder_name: folder where files named after one epitope are found, they contain labeled tcr chains
    @param tcr_chains: the different chains we want in our data frame
    @return: a data frame
    """
    dfs = []
    for epitope in epitopes:
        df = load_epitope_tcr_data(folder_name, epitope, tcr_chains)
        dfs.append(df)

    df = pd.concat(dfs)
    return df


def calculate_imap_shape(df_train: pandas.DataFrame, tcr_chain: str):
    """
    finds the  maximum epitope and tcr chain lengths
    """
    train_epitopes = df_train['Epitope'].tolist()
    train_tcr = df_train[tcr_chain].tolist()

    return len(max(train_tcr, key=len)), len(max(train_epitopes, key=len))


def generate_interaction_map(tcr_chain, epitope, features_string, operator_string):
    # specify the different interaction map features and the operator that is used to calculate the entries
    features_list = parse_features(features_string)
    operator = parse_operator(operator_string)
    feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)

    return feature_builder.generate_peptides_feature(tcr_chain, epitope)


def pad(interaction_map, target_height, target_width):
    height_padding = target_height - interaction_map.shape[0]
    padding_top = height_padding // 2
    padding_bot = height_padding - padding_top

    width_padding = target_width - interaction_map.shape[1]
    padding_left = width_padding // 2
    padding_right = width_padding - padding_left

    padding = [
        (padding_top, padding_bot),
        (padding_left, padding_right),
        (0, 0)
    ]

    return np.pad(interaction_map, padding, mode='constant', constant_values=0)


def generate_padded_imap(tcr_chain, epitope, max_len_tcr, max_len_epitope):
    imap = pad(
        generate_interaction_map(
            tcr_chain,
            epitope,
            'hydrophob,isoelectric,mass,hydrophil',
            'absdiff'
        ),
        max_len_tcr,
        max_len_epitope
    )
    return imap


def add_imaps_and_relabel(df, chain_name, max_len_tcr, max_len_epitope):
    # for each tcr-epitope pair, generate an interaction map, zero pad it and store it in df
    imaps = []
    for index, row, in df.iterrows():
        imap = generate_padded_imap(
            tcr_chain=row[chain_name],
            epitope=row['Epitope'],
            max_len_tcr=max_len_tcr,
            max_len_epitope=max_len_epitope
        )
        imaps.append(imap)

    df['interaction_map'] = imaps
    df = df[['interaction_map', 'Label']]
    return df


def generate_imap_dataset(train_folder, tcr_chains, shape=None):
    chain_name = tcr_chains[0] if len(tcr_chains) == 0 else ''.join(tcr_chains)

    epitopes = list_epitopes(train_folder)
    df = load_complete_data(epitopes, train_folder, tcr_chains)

    max_len_tcr, max_len_epitope = calculate_imap_shape(df, chain_name)

    if shape:
        max_len_tcr = shape[0]
        max_len_epitope = shape[1]

    df = add_imaps_and_relabel(df, chain_name, max_len_tcr, max_len_epitope)
    imap_shape = max_len_tcr, max_len_epitope, 4
    return df, imap_shape
