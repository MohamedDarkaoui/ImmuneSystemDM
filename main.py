from src.data.data_processing import generate_imap_dataset, generate_test_data, InteractionMapMode
from src.models.DualInputModel import DualInputModel
import keras_tuner
import tensorflow as tf
from sklearn.model_selection import train_test_split

TRAIN_FOLDER = 'data/training_data/'
TEST_FOLDER = 'data/true_set/'
MODEL_PATH = 'models/TEST/4DIM'
ALPHA = ['TRA_CDR3']
BETA = ['TRB_CDR3']
CHAINS = ['TRA_CDR3', 'TRB_CDR3']
SAVE_PATH = 'models/TEST2/combine/model3'
MODE = InteractionMapMode.SINGLE
DUAL_PATH = 'models/dual/model3'

df_alpha, shape_alpha = generate_imap_dataset(TRAIN_FOLDER, ALPHA, MODE)
df_beta, shape_beta = generate_imap_dataset(TRAIN_FOLDER, BETA, MODE)


def build_model(hp):
    model = DualInputModel(
        save_path=SAVE_PATH,
        input_shape=[shape_alpha, shape_beta],
        kernel_size=hp.Choice('kernel_size', values=[3, 5]),
        l2=hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log'),
        nr_filters_layer1=hp.Int('nr_filters_layer1', min_value=32, max_value=256, step=32),
        nr_filters_layer2=hp.Int('nr_filters_layer2', min_value=32, max_value=256, step=32),
        second_unit=hp.Boolean('second_unit'),
        nr_dense=hp.Int('nr_dense', min_value=16, max_value=64, step=16),
        optimizer='rmsprop',
        batch_size=hp.Int('batch_size', min_value=16, max_value=64, step=16),
        pool_size=hp.Choice('pool_size', values=[2, 3]),
        dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.25)
    )
    return model.model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=100,
    executions_per_trial=1,
    overwrite=True,
    directory="tuning_results",
    project_name="dual_model",
)

X_alpha_train, X_alpha_val, X_beta_train, X_beta_val, y_train, y_val = train_test_split(
    df_alpha['interaction_map'].tolist(),
    df_beta['interaction_map'].tolist(),
    df_alpha['Label'].tolist(),
    test_size=0.2,
    random_state=42)
train_data = tf.data.Dataset.from_tensor_slices(((X_alpha_train, X_beta_train), y_train))
val_data = tf.data.Dataset.from_tensor_slices(((X_alpha_val, X_beta_val), y_val))

train_data = train_data.shuffle(buffer_size=len(X_alpha_train)).batch(32)
val_data = val_data.batch(32)

print(tuner.search_space_summary())
tuner.search(train_data, epochs=2, validation_data=val_data)
