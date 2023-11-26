import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data.data_processing import generate_imap_dataset
from src.models.CNN_model.model import build_model
from src.visualization.plotting import *

# train_folder = '../../../../data/training_data/'
# test_folder = '../../../../data/true_set/'
# image_loss_path = '../../../visualization/img/trb_model_loss'
# image_metrics_path = '../../../visualization/img/trb_model_metrics'
# model_path = '../../../../models/binary_classification_tcrb_30_epochs'
# tcr_chains = ['TRB_CDR3']


def train(
        imap_df,
        imap_shape,
        model_path,
        epochs,
        image_loss_path=None,
        image_metrics_path=None
):
    model = build_model(imap_shape)

    # split data into train and validation set
    X = list(imap_df['interaction_map'])
    y = list(imap_df['Label'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # convert data to tf.data.Dataset objects and batch
    train_data = tf.data.Dataset.from_tensor_slices((np.array(X_train), np.array(y_train)))
    val_data = tf.data.Dataset.from_tensor_slices((np.array(X_val), np.array(y_val)))

    train_data = train_data.shuffle(buffer_size=len(train_data), seed=42, reshuffle_each_iteration=True).batch(32)
    val_data = val_data.batch(32)

    history = model.fit(
        x=train_data,
        epochs=epochs,
        validation_data=val_data,
        class_weight=None,
        max_queue_size=2,
        use_multiprocessing=True,
        verbose=1,
        shuffle=False,
    )

    model.save(model_path)
    return history
