import os

import pandas
import pandas as pd
import numpy as np
from src.external.bio.peptide_feature import parse_features, parse_operator
from src.external.bio.feature_builder import CombinedPeptideFeatureBuilder
from copy import deepcopy
from enum import Enum
from tqdm import tqdm


class InteractionMapMode(Enum):
    SINGLE = 1
    CONCATENATE = 2
    COMBINE = 3
    MERGE_DIMENSIONAL = 4
    DUAL_INPUT = 5


class TCR_info:
    def __init__(self, chain, height, width):
        self.chain = chain
        self.height = height
        self.width = width


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


def load_epitope_tcr_data(folder_name: str, epitope_name: str, tcr_chains: list):
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

    df = pd.concat(dfs, ignore_index=True)
    return df


def calculate_imap_shape(df_train: pandas.DataFrame, tcr_chains: list, mode: InteractionMapMode):
    """
    finds the  maximum epitope and tcr chain lengths
    """

    epitopes = df_train['Epitope'].unique().tolist()

    height = 0
    width = len(max(epitopes, key=len))
    depth = 4

    if mode.value == InteractionMapMode.COMBINE.value or mode.value == InteractionMapMode.MERGE_DIMENSIONAL.value:
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
    imaps = []
    for chain in tcr_chains:
        features_list = parse_features(features_string)
        operator = parse_operator(operator_string)
        feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)
        imap = feature_builder.generate_peptides_feature(chain, epitope)
        imaps.append(imap)
    return imaps


def generate_interaction_maps_and_pad(tcr_chains, epitope, features_string='hydrophob,isoelectric,mass,hydrophil'
                                      , operator_string='absdiff'):
    imaps = []
    for tcr_info in tcr_chains:
        features_list = parse_features(features_string)
        operator = parse_operator(operator_string)
        feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)
        imap = feature_builder.generate_peptides_feature(tcr_info.chain, epitope)
        imap = pad(imap, tcr_info.height, tcr_info.width)
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
        if mode.value == InteractionMapMode.COMBINE.value:
            imap = combine_imaps(imap)
            imaps.append(imap)
        elif mode.value == InteractionMapMode.MERGE_DIMENSIONAL.value:
            combined_imap = np.zeros((2, height, width, 4))
            combined_imap[0, :, :, :] = imap[0]
            combined_imap[1, :, :, :] = imap[1]
            imaps.append(combined_imap)
        else:
            imaps.append(imap[0])
    df['interaction_map'] = imaps
    df = df[['interaction_map', 'Label', 'Epitope']]
    return df


def add_imaps_column(df, tcr_chains, height, width, mode: InteractionMapMode):
    # for each tcr-epitope pair, generate an interaction map, zero pad it and store it in df
    imaps = []
    for index, row, in df.iterrows():
        imap = generate_interaction_maps_and_pad(
            tcr_chains=[row[tcr_chains[i]] for i in range(len(tcr_chains))],
            epitope=row['Epitope'],
            height=height,
            width=width
        )
        if mode.value == InteractionMapMode.DUAL_INPUT.value:
            assert len(imap) == 2
            imaps.append(imap)

    if imaps and isinstance(imaps[0], list):
        alphas, betas = zip(*((imap[0], imap[1]) for imap in imaps))
        df['TRA_imap'] = alphas
        df['TRB_imap'] = betas

    return df


def generate_imap_dataset(train_folder: str, tcr_chains: list, mode: InteractionMapMode, shape: tuple = None):
    epitopes = list_epitopes(train_folder)
    df = load_complete_data(epitopes, train_folder, tcr_chains)

    if mode.value == InteractionMapMode.CONCATENATE.value:
        df, tcr_chains = concat_columns(df, tcr_chains)
    height, width, depth = shape if shape else calculate_imap_shape(df, tcr_chains, mode=mode)

    df = add_imaps_and_relabel(df, tcr_chains, height, width, mode)
    imap_shape = df.iloc[0, 0].shape
    return df, imap_shape


def generate_test_data(test_folder: str, tcr_chains: list, mode: InteractionMapMode, imap_shape: tuple):
    epitopes = list_epitopes(test_folder)
    data = []
    for epitope in epitopes:
        chains = deepcopy(tcr_chains)
        df = load_epitope_tcr_data(test_folder, epitope, chains)
        if mode.value == InteractionMapMode.CONCATENATE.value:
            df, chains = concat_columns(df, chains)

        height, width = imap_shape[0], imap_shape[1]
        if mode.value == InteractionMapMode.MERGE_DIMENSIONAL.value:
            height, width = imap_shape[1], imap_shape[2]
        df = add_imaps_and_relabel(df, chains, height, width, mode)
        data.append((epitope, df['interaction_map'].tolist(), df['Label'].tolist()))
    return data


def generate_ranking_test_data(test_folder: str, tcr_chains: list, mode: InteractionMapMode, imap_shape: tuple):
    shape_alpha, shape_beta = imap_shape
    height_alpha, width_alpha = shape_alpha[0], shape_alpha[1]
    height_beta, width_beta = shape_beta[0], shape_beta[1]
    epitopes = list_epitopes(test_folder)
    data = []

    for epitope in tqdm(epitopes):
        df = load_epitope_tcr_data(test_folder, epitope, tcr_chains)
        df = df[df['Label'] == 1]
        dfs = [df]
        for epitope2 in epitopes:
            if epitope == epitope2:
                continue
            df2 = df.copy()
            df2['Epitope'] = epitope2
            df2['Label'] = 0
            dfs.append(df2)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.drop_duplicates(inplace=True)
        alphas, betas = [], []
        for index, row in final_df.iterrows():
            imap_alpha, imap_beta = generate_interaction_maps_and_pad(
                [
                    TCR_info(row['TRA_CDR3'], height_alpha, width_alpha),
                    TCR_info(row['TRB_CDR3'], height_beta, width_beta)
                ],
                row['Epitope']
            )
            alphas.append(imap_alpha)
            betas.append(imap_beta)

        final_df['imap_alpha'] = alphas
        final_df['imap_beta'] = betas
        final_df['TRA_TRB_CDR3'] = final_df['TRA_CDR3'] + "_" + final_df['TRB_CDR3']
        dfs_list = [(key, group[['imap_alpha', 'imap_beta', 'Label']]) for key, group in final_df.groupby('TRA_TRB_CDR3')]
        keys, groups = zip(*dfs_list)
        alpha_groups = [group['imap_alpha'].to_list() for group in groups]
        beta_groups = [group['imap_beta'].tolist() for group in groups]
        labels = [group['Label'].tolist() for group in groups]
        data.append(
            {
                'Epitope': epitope,
                'tcrs': keys,
                'alpha_imaps': alpha_groups,
                'beta_imaps': beta_groups,
                'labels': labels
            }
        )

    return data
