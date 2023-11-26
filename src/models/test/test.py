import tensorflow as tf
from src.data.data_processing import *
from sklearn.metrics import roc_auc_score

# train_folder = '../../../data/training_data/'
# test_folder = '../../../data/true_set/'
# tcr_chains = ['TRB_CDR3']


def evaluate(model_path, test_folder, tcr_chains, imap_shape):
    test_epitopes = list_epitopes(test_folder)
    loaded_model = tf.keras.models.load_model(model_path)

    chain_name = ''.join(tcr_chains) if len(tcr_chains) > 1 else tcr_chains[0]

    auc_scores = dict()
    max_len_tcr, max_len_epitope = imap_shape[0], imap_shape[1]

    for test_epitope in test_epitopes:
        df = load_epitope_tcr_data(test_folder, test_epitope, tcr_chains)
        df = add_imaps_and_relabel(df, chain_name, max_len_tcr, max_len_epitope)

        print('shape', df.iloc[0, 0].shape)
        imaps = list(df['interaction_map'])
        imaps_tensor = tf.stack(imaps)
        true_labels = list(df['Label'])
        predicted_scores = loaded_model.predict(imaps_tensor)
        auc = roc_auc_score(true_labels, predicted_scores)
        auc_scores[test_epitope] = auc
    return auc_scores
