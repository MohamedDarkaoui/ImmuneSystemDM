from src.data.data_processing import generate_imap_dataset, generate_test_data, InteractionMapMode
from src.models.DualInputModel import DualInputModel
from src.visualization.plotting import plot_metrics, plot_dict
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


# def build_model(hp):
#     model = DualInputModel(
#         save_path=SAVE_PATH,
#         input_shape=[shape_alpha, shape_beta],
#         kernel_size=hp.Choice('kernel_size', values=[3, 5]),
#         l2=hp.Float('l2', min_value=1e-5, max_value=1e-2, sampling='log'),
#         nr_filters_layer1=hp.Int('nr_filters_layer1', min_value=32, max_value=256, step=32),
#         nr_filters_layer2=hp.Int('nr_filters_layer2', min_value=32, max_value=256, step=32),
#         second_unit=hp.Boolean('second_unit'),
#         nr_dense=hp.Int('nr_dense', min_value=16, max_value=64, step=16),
#         optimizer='rmsprop',
#         batch_size=hp.Int('batch_size', min_value=16, max_value=64, step=16),
#         pool_size=hp.Choice('pool_size', values=[2, 3]),
#         dropout_rate=hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.25)
#     )
#     return model.model
#
# tuner = keras_tuner.RandomSearch(
#     hypermodel=build_model,
#     objective="val_loss",
#     max_trials=100,
#     executions_per_trial=1,
#     overwrite=True,
#     directory="tuning_results",
#     project_name="dual_model",
# )
#
# X_alpha_train, X_alpha_val, X_beta_train, X_beta_val, y_train, y_val = train_test_split(
#     df_alpha['interaction_map'].tolist(),
#     df_beta['interaction_map'].tolist(),
#     df_alpha['Label'].tolist(),
#     test_size=0.2,
#     random_state=42)
# train_data = tf.data.Dataset.from_tensor_slices(((X_alpha_train, X_beta_train), y_train))
# val_data = tf.data.Dataset.from_tensor_slices(((X_alpha_val, X_beta_val), y_val))
#
# train_data = train_data.shuffle(buffer_size=len(X_alpha_train)).batch(32)
# val_data = val_data.batch(32)
#
# print(tuner.search_space_summary())
# tuner.search(train_data, epochs=2, validation_data=val_data)

#
# from src.data.data_processing import *
# model = DualInputModel(
#     second_unit=True,
#     save_path=DUAL_PATH,
#     input_shape=[shape_alpha, shape_beta],
#     l2=3e-05,
#     kernel_size=3,
#     nr_filters_layer1=256,
#     nr_filters_layer2=256,
#     nr_dense=64,
#     optimizer='sgd',
#     pool_size=2,
#     dropout_rate=0.25
# )
# # history = model.train(imap_df=(df_alpha, df_beta), epochs=100)
#
# test_data = (
#     generate_test_data(test_folder=TEST_FOLDER, tcr_chains=ALPHA, mode=MODE, imap_shape=shape_alpha),
#     generate_test_data(test_folder=TEST_FOLDER, tcr_chains=BETA, mode=MODE, imap_shape=shape_beta)
# )
#
# # auc_scores = model.evaluate(test_data=test_data)
# # print(sum(auc_scores.values()) / len(auc_scores))
# # plot_auc(auc_scores)
#
# ranking_data = generate_ranking_test_data(TEST_FOLDER, CHAINS, InteractionMapMode.DUAL_INPUT, (shape_alpha, shape_beta))
#
#
# ranks = model.evaluate_rank(ranking_data)
#
# print(ranks)
# vals = ranks.values()
# print("avg: ", sum(vals)/len(vals))



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the sequences
alpha = "CAFMSLYGGSQGNLIF"
beta = "CASSYPYRGLLAGSGNTIYF"
epitope = "ATDALMTGF"

# Concatenate alpha and beta sequences
concatenated_sequence = alpha + beta

# Generate random values for the heatmap
np.random.seed(0)  # For reproducibility
interaction_map = np.random.rand(len(beta), len(epitope))

# Add padding to the interaction map
padding = 2
padded_map = np.full((len(beta) + 2 * padding, len(epitope) + 2 * padding), np.nan)
padded_map[padding:padding + len(beta), padding:padding + len(epitope)] = interaction_map

# Create the heatmap with the 'Blues' colormap
fig, ax = plt.subplots(figsize=(25, 25 / 3))  # Increased figure size for better readability
sns.heatmap(padded_map, cmap='Blues', cbar=False,
            xticklabels=['']*padding + list(epitope) + ['']*padding,
            yticklabels=['']*padding + list(beta) + ['']*padding,
            ax=ax, linewidths=0.5, linecolor='lightgray')

# Set axis labels and title
ax.set_xlabel('Epitope')
ax.set_ylabel('Concatenated Alpha + Beta')
ax.set_title('Mass')

# Ensure each cell is rectangular
ax.set_aspect(0.5)  # Adjust this value to get the desired aspect ratio

plt.xticks(rotation=0)  # Keep x-axis labels horizontal for better readability
plt.yticks(rotation=0)  # Ensure y-axis labels are horizontal

plt.show()
