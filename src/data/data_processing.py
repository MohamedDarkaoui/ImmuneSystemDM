import os
import pandas as pd
import numpy as np
from src.external.bio.peptide_feature import parse_features, parse_operator
from src.external.bio.feature_builder import CombinedPeptideFeatureBuilder


def list_epitopes(folder_name):
    files = os.listdir(folder_name)
    epitopes = [
        file.replace('.txt', '') for file in files if file != 'README.txt' and not file.startswith('test')
    ]
    return epitopes


def load_epitope_trb_data(folder_name,  epitope_name):
    df = pd.read_csv(folder_name + epitope_name + '.txt', sep='\t')
    df = df[['TRB_CDR3', 'Label']]
    df['Label'] = df['Label'].replace(-1, 0)
    return df


def read_all_trb_sequences(folder_name, epitope_name):
    df = load_epitope_trb_data(folder_name, epitope_name)
    trb_sequences = df['TRB_CDR3'].tolist()
    return trb_sequences


def max_trb_epitope_lengths(train_folder, test_folder):
    train_epitopes = list_epitopes(train_folder)
    test_epitopes = list_epitopes(test_folder)

    all_epitopes = train_epitopes + test_epitopes
    all_trb = []

    for epitope in train_epitopes:
        trb_sequences = read_all_trb_sequences(train_folder, epitope)
        all_trb += trb_sequences

    for epitope in test_epitopes:
        trb_sequences = read_all_trb_sequences(test_folder, epitope)
        all_trb += trb_sequences

    return len(max(all_trb, key=len)), len(max(all_epitopes, key=len))


def generate_interaction_map(trb, epitope, features_string, operator_string):
    features_list = parse_features(features_string)
    operator = parse_operator(operator_string)
    feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)
    return feature_builder.generate_peptides_feature(trb, epitope)


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
