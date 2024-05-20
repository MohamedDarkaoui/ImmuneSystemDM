from src.models.TCR3DModel import TCR3DModel
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os


class DualInputModel(TCR3DModel):
    """
    Expects two inputs, alpha chain and beta chain separately
    """

    def __init__(
            self,
            save_path,
            input_shape,
            kernel_size=3,
            l2=0.01,
            nr_filters_layer1=128,
            nr_filters_layer2=64,
            nr_dense=32,
            dense_activation='relu',
            optimizer='rmsprop',
            batch_size=32,
            pool_size=2,
            dropout_rate=0.25,
            second_unit=False
    ):
        self.second_unit = second_unit
        super().__init__(save_path, input_shape, kernel_size, l2, nr_filters_layer1, nr_filters_layer2, nr_dense,
                         dense_activation, optimizer, batch_size, pool_size, dropout_rate)

    def create_unit(self, x, is_first):
        if is_first:
            x = self.create_conv(self.nr_filters_layer1, input_shape=self.input_shape)(x)
        else:
            x = self.create_conv(self.nr_filters_layer1)(x)
        x = keras.layers.BatchNormalization()(x)
        x = self.create_conv(self.nr_filters_layer2)(x)
        x = self.max_pool()(x)
        x = self.spacial_dropout()(x)
        x = keras.layers.BatchNormalization()(x)
        return x

    def create_branch(self, input_layer):
        # First unit
        x = self.create_unit(input_layer, True)

        # Second unit
        if self.second_unit:
            x = self.create_unit(x, False)

        # # Flatten and dense layers
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(self.nr_dense, activation=self.dense_activation)(x)
        # x = keras.layers.GlobalAveragePooling2D()(x)
        return x

    def build_model(self):
        alpha_shape = self.input_shape[0]
        beta_shape = self.input_shape[1]

        alpha_input = keras.layers.Input(shape=alpha_shape)
        beta_input = keras.layers.Input(shape=beta_shape)

        alpha_branch = self.create_branch(input_layer=alpha_input)

        beta_branch = self.create_branch(input_layer=beta_input)

        merged = keras.layers.concatenate([alpha_branch, beta_branch])

        x = keras.layers.Dense(self.nr_filters_layer1, activation='relu')(merged)
        # x = keras.layers.BatchNormalization()(x)

        output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs=[alpha_input, beta_input], outputs=output)

        # print(model.summary())
        return model

    def train(self, imap_df, epochs, image_loss_path=None, image_metrics_path=None):
        alpha_df, beta_df = imap_df
        assert len(alpha_df) == len(beta_df), "Alpha and Beta DataFrames must be of the same length"
        X_alpha = np.array(alpha_df['interaction_map'].tolist())
        X_beta = np.array(beta_df['interaction_map'].tolist())
        y = np.array(alpha_df['Label'].tolist())

        X_alpha_train, X_alpha_val, X_beta_train, X_beta_val, y_train, y_val = train_test_split(
            X_alpha, X_beta, y, test_size=0.2, random_state=42)

        train_data = tf.data.Dataset.from_tensor_slices(((X_alpha_train, X_beta_train), y_train))
        val_data = tf.data.Dataset.from_tensor_slices(((X_alpha_val, X_beta_val), y_val))

        train_data = train_data.shuffle(buffer_size=len(X_alpha_train),
                                        reshuffle_each_iteration=True).batch(self.batch_size)
        val_data = val_data.batch(self.batch_size)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.save_path),
            monitor='val_loss', save_best_only=True)

        history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=[early_stopping, model_checkpoint],
            use_multiprocessing=True,
            verbose=1)

        self.is_trained = True
        return history

    from sklearn.metrics import roc_auc_score
    import numpy as np

    def evaluate(self, test_data):
        model = self.model if self.is_trained else tf.keras.models.load_model(self.save_path)

        all_true_labels = []
        all_predicted_scores = []
        auc_scores = dict()

        test_data_alpha, test_data_beta = test_data
        for i in range(len(test_data_alpha)):
            epitope, imaps_alpha, true_labels = test_data_alpha[i]
            epitope2, imaps_beta, true_labels2 = test_data_beta[i]
            assert (epitope, true_labels) == (epitope2, true_labels2)

            imaps_alpha_tensor = tf.stack(imaps_alpha)
            imaps_beta_tensor = tf.stack(imaps_beta)
            predicted_scores = model.predict([imaps_alpha_tensor, imaps_beta_tensor])

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
            alpha_imaps_list = group['alpha_imaps']
            beta_imaps_list = group['beta_imaps']
            labels_list = group['labels']
            group_ranks = []
            print(epitope)
            for i in range(len(tcrs)):
                alpha_imaps = alpha_imaps_list[i]
                beta_imaps = beta_imaps_list[i]
                labels = labels_list[i]
                assert labels[0] == 1
                assert labels.count(1) == 1
                alpha_imaps_tensor = tf.stack(alpha_imaps)
                beta_imaps_tensor = tf.stack(beta_imaps)
                prediction_scores = model.predict([alpha_imaps_tensor, beta_imaps_tensor])
                prediction_scores = prediction_scores.tolist()
                rank = len([pred for pred in prediction_scores if pred > prediction_scores[0]]) + 1
                group_ranks.append(rank)

            ranks[epitope] = np.mean(group_ranks)
        return ranks
