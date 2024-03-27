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
        threshold = 0.5
        class_labels = (predicted_scores >= threshold).astype(int)
        print(predicted_scores)
        macro_auc_01 = roc_auc_score(true_labels, predicted_scores, max_fpr=0.1, average='macro')
        auc_scores[epitope] = macro_auc_01
    return auc_scores
