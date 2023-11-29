import tensorflow as tf
from src.data.data_processing import *
from sklearn.metrics import roc_auc_score

# train_folder = '../../../data/training_data/'
# test_folder = '../../../data/true_set/'
# tcr_chains = ['TRB_CDR3']


def evaluate(model_path, test_data):
    loaded_model = tf.keras.models.load_model(model_path)
    auc_scores = dict()

    for test_epitope in test_data:
        epitope, imaps, true_labels = test_epitope

        print('shape', imaps[0].shape)
        imaps_tensor = tf.stack(imaps)
        predicted_scores = loaded_model.predict(imaps_tensor)
        auc = roc_auc_score(true_labels, predicted_scores)
        auc_scores[epitope] = auc
    return auc_scores
