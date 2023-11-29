import os

import pandas
import pandas as pd
import numpy as np
from src.external.bio.peptide_feature import parse_features, parse_operator
from src.external.bio.feature_builder import CombinedPeptideFeatureBuilder
from copy import deepcopy
from enum import Enum


class InteractionMapMode(Enum):
    SINGLE = 1
    CONCATENATE = 2
    COMBINE = 3


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
    return df, [col_name]


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


def calculate_imap_shape(df_train: pandas.DataFrame, tcr_chains: list, mode: InteractionMapMode):
    """
    finds the  maximum epitope and tcr chain lengths
    """

    epitopes = df_train['Epitope'].unique().tolist()

    height = 0
    width = len(max(epitopes, key=len))
    depth = 4

    if mode.value == InteractionMapMode.COMBINE.value:
        assert len(tcr_chains) > 1

        for chain in tcr_chains:
            tcr_list = df_train[chain].unique().tolist()
            new_height = len(max(tcr_list, key=len))
            if new_height > height:
                height = new_height
            depth = 8
    elif mode.value == InteractionMapMode.CONCATENATE.value:
        assert len(tcr_chains) == 1
        tcr_chain = tcr_chains[0]
        tcr_list = df_train[tcr_chain].unique().tolist()
        height = len(max(tcr_list, key=len))
    else:
        assert len(tcr_chains) == 1
        tcr_list = df_train[tcr_chains[0]].unique().tolist()
        height = len(max(tcr_list, key=len))

    return height, width, depth


def generate_interaction_maps(tcr_chains, epitope, features_string, operator_string):
    # specify the different interaction map features and the operator that is used to calculate the entries
    imaps = []
    for chain in tcr_chains:
        features_list = parse_features(features_string)
        operator = parse_operator(operator_string)
        feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)
        imap = feature_builder.generate_peptides_feature(chain, epitope)
        imaps.append(imap)
    return imaps


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


def generate_padded_imaps(tcr_chains, epitope, height, width):
    imaps = generate_interaction_maps(
        tcr_chains,
        epitope,
        'hydrophob,isoelectric,mass,hydrophil',
        'absdiff'
    )
    padded_imaps = []
    for i in range(len(tcr_chains)):
        imap = pad(
            imaps[i],
            height,
            width
        )
        padded_imaps.append(imap)
    return padded_imaps


def combine_imaps(padded_imaps: list):
    return np.concatenate(padded_imaps, axis=2)


def add_imaps_and_relabel(df, tcr_chains, height, width, mode: InteractionMapMode):
    # for each tcr-epitope pair, generate an interaction map, zero pad it and store it in df
    imaps = []
    for index, row, in df.iterrows():
        imap = generate_padded_imaps(
            tcr_chains=[row[tcr_chains[i]] for i in range(len(tcr_chains))],
            epitope=row['Epitope'],
            height=height,
            width=width
        )
        if mode == InteractionMapMode.COMBINE:
            imap = combine_imaps(imap)
            imaps.append(imap)
        else:
            imaps.append(imap[0])
    df = df.reset_index(drop=True)
    df['interaction_map'] = imaps
    df = df[['interaction_map', 'Label']]

    return df


def generate_imap_dataset(train_folder: str, tcr_chains: list, mode: InteractionMapMode, shape: tuple = None):
    epitopes = list_epitopes(train_folder)
    df = load_complete_data(epitopes, train_folder, tcr_chains)

    if mode.value == InteractionMapMode.CONCATENATE.value:
        df, tcr_chains = concat_columns(df, tcr_chains)
    height, width, depth = shape if shape else calculate_imap_shape(df, tcr_chains, mode=mode)

    df = add_imaps_and_relabel(df, tcr_chains, height, width, mode)
    imap_shape = height, width, depth
    return df, imap_shape


def generate_test_data(test_folder: str, tcr_chains: list, mode: InteractionMapMode, imap_shape: tuple):
    epitopes = list_epitopes(test_folder)
    data = []
    for epitope in epitopes:
        chains = deepcopy(tcr_chains)
        df = load_epitope_tcr_data(test_folder, epitope, chains)
        if mode == InteractionMapMode.CONCATENATE:
            df, chains = concat_columns(df, chains)
        df = add_imaps_and_relabel(df, chains, imap_shape[0], imap_shape[1], mode)
        data.append((epitope, df['interaction_map'].tolist(), df['Label'].tolist()))
    return data
