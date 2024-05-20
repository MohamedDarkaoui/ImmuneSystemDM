import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class AbstractModel:
    def __init__(self,
                 save_path,
                 input_shape,
                 nr_filters_layer1=128,
                 nr_filters_layer2=64,
                 nr_dense=32,
                 dense_activation='relu',
                 optimizer='rmsprop',
                 batch_size=32
                 ):
        self.save_path = save_path
        self.input_shape = input_shape
        self.nr_filters_layer1 = nr_filters_layer1
        self.nr_filters_layer2 = nr_filters_layer2
        self.nr_dense = nr_dense
        self.dense_activation = dense_activation
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.model = self.build_model()
        self.compile_model()
        self.is_trained = False

    def create_branch(self, input_layer):
        # First unit
        x = self.create_conv(self.nr_filters_layer1, input_shape=self.input_shape)(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = self.create_conv(self.nr_filters_layer2)(x)
        x = self.max_pool()(x)
        x = self.spacial_dropout()(x)
        x = keras.layers.BatchNormalization()(x)

        # Second unit
        x = self.create_conv(self.nr_filters_layer1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = self.create_conv(self.nr_filters_layer2)(x)
        x = self.max_pool()(x)
        x = self.spacial_dropout()(x)
        x = keras.layers.BatchNormalization()(x)

        # Flatten and dense layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self.nr_dense, activation=self.dense_activation)(x)
        output_layer = keras.layers.Dense(1, activation='sigmoid')(x)
        return output_layer

    def build_model(self):
        input_layer = keras.layers.Input(shape=self.input_shape)
        output_layer = self.create_branch(input_layer)

        # Model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        print(model.summary())
        return model

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=keras.losses.BinaryCrossentropy(),
                           metrics=[keras.metrics.AUC(curve="ROC", name="roc_auc")]
                           )

    def create_conv(self, nr_filters: int, **kwargs):
        raise NotImplementedError("Not implemented")

    def max_pool(self):
        raise NotImplementedError("Not implemented")

    def spacial_dropout(self):
        raise NotImplementedError("Not implemented")

    def train(self, imap_df, epochs, image_loss_path=None, image_metrics_path=None):
        # split data into train and validation set
        X = imap_df['interaction_map'].tolist()
        y = imap_df['Label'].tolist()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # convert data to tf.data.Dataset objects and batch
        train_data = tf.data.Dataset.from_tensor_slices((np.array(X_train), np.array(y_train)))
        val_data = tf.data.Dataset.from_tensor_slices((np.array(X_val), np.array(y_val)))

        train_data = train_data.shuffle(buffer_size=len(train_data), seed=42, reshuffle_each_iteration=True).batch(32)
        val_data = val_data.batch(self.batch_size)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.save_path,
            monitor='val_loss',
            save_best_only=True
        )

        history = self.model.fit(
            x=train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[early_stopping, model_checkpoint],
            class_weight=None,
            max_queue_size=2,
            use_multiprocessing=True,
            verbose=1,
            shuffle=False,
        )

        self.is_trained = True
        return history

    def evaluate(self, test_data):
        model = self.model if self.is_trained else tf.keras.models.load_model(self.save_path)
        auc_scores = dict()

        all_true_labels = []
        all_predicted_scores = []

        for test_epitope in test_data:
            epitope, imaps, true_labels = test_epitope

            imaps_tensor = tf.stack(imaps)
            predicted_scores = model.predict(imaps_tensor)

            all_true_labels.extend(true_labels)
            all_predicted_scores.extend(predicted_scores)

            auc = roc_auc_score(true_labels, predicted_scores)
            auc_scores[epitope] = auc

        all_true_labels = np.array(all_true_labels)
        all_predicted_scores = np.array(all_predicted_scores)
        micro_auc = roc_auc_score(all_true_labels, all_predicted_scores)

        macro_auc = sum(auc_scores.values()) / len(auc_scores)

        return auc_scores, macro_auc, micro_auc

    def evaluate_rank(self, test_data: list):
        model = self.model if self.is_trained else tf.keras.models.load_model(self.save_path)

        ranks = dict()
        for group in test_data:
            epitope = group['Epitope']
            tcrs = group['tcrs']
            imaps_list = group['interaction_maps']
            labels_list = group['labels']
            group_ranks = []
            print(epitope)
            for i in range(len(tcrs)):
                imaps = imaps_list[i]
                labels = labels_list[i]
                assert labels[0] == 1
                assert labels.count(1) == 1
                imaps_tensor = tf.stack(imaps)
                prediction_scores = model.predict(imaps_tensor)

                prediction_scores = prediction_scores.tolist()
                rank = len([pred for pred in prediction_scores if pred > prediction_scores[0]]) + 1
                group_ranks.append(rank)

            ranks[epitope] = np.mean(group_ranks)
        return ranks


