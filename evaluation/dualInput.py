import os
import numpy as np
from src.data.data_processing import generate_imap_dataset, generate_test_data, generate_ranking_test_data, InteractionMapMode
from src.models.DualInputModel import DualInputModel
from src.visualization.plotting import plot_dict

def evaluate_dual_input_models_in_directory(models_dir, train_folder, test_folder, alpha_chains, beta_chains, mode, l2=0.00001):
    df_alpha, shape_alpha = generate_imap_dataset(train_folder, alpha_chains, InteractionMapMode.SINGLE)
    df_beta, shape_beta = generate_imap_dataset(train_folder, beta_chains, InteractionMapMode.SINGLE)

    test_data = (
        generate_test_data(test_folder=test_folder, tcr_chains=alpha_chains, mode=mode, imap_shape=shape_alpha),
        generate_test_data(test_folder=test_folder, tcr_chains=beta_chains, mode=mode, imap_shape=shape_beta)
    )

    all_auc_scores = []
    all_macro_auc = []
    all_micro_auc = []

    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f"Found {len(model_dirs)} model directories in {models_dir}")

    for model_dir in model_dirs:
        model_path = os.path.join(models_dir, model_dir)

        print(f"Evaluating model in directory: {model_dir}")

        model = DualInputModel(
            second_unit=True,
            save_path=model_path,
            input_shape=[shape_alpha, shape_beta],
            l2=l2,
            kernel_size=3,
            nr_filters_layer1=256,
            nr_filters_layer2=256,
            nr_dense=64,
            optimizer='sgd',
            pool_size=2,
            dropout_rate=0.25
        )

        try:
            auc_scores, macro_auc, micro_auc = model.evaluate(test_data=test_data)
            all_auc_scores.append(auc_scores)
            all_macro_auc.append(macro_auc)
            all_micro_auc.append(micro_auc)
        except Exception as e:
            print(f"Failed to evaluate model in directory {model_dir}: {e}")

    if all_macro_auc and all_micro_auc:
        average_macro_auc = np.mean(all_macro_auc)
        average_micro_auc = np.mean(all_micro_auc)
        print("Average Macro AUC:", average_macro_auc)
        print("Average Micro AUC:", average_micro_auc)
        epitope_keys = all_auc_scores[0].keys()
        average_per_epitope_auc = {key: np.mean([scores[key] for scores in all_auc_scores]) for key in epitope_keys}
        print("Average Per-Epitope AUC Scores:", average_per_epitope_auc)
        plot_dict(average_per_epitope_auc, 'Epitope', 'Average AUC', 'Average AUC for Different Epitopes', average_line=True)
    else:
        print("No models were successfully evaluated.")

    ranking_data = generate_ranking_test_data(test_folder, alpha_chains + beta_chains, mode, (shape_alpha, shape_beta))

    # Evaluate ranking for the last loaded model (if any)
    if model:
        ranks = model.evaluate_rank(ranking_data)
        print("Ranks:", ranks)
        vals = ranks.values()
        print("Average Rank:", sum(vals) / len(vals))
    else:
        print("No valid model available for ranking evaluation.")

TRAIN_FOLDER = '../data/training_data/'
TEST_FOLDER = '../data/true_set/'
MODELS_DIR = '../models/dual'
ALPHA = ['TRA_CDR3']
BETA = ['TRB_CDR3']
CHAINS = ['TRA_CDR3', 'TRB_CDR3']
MODE = InteractionMapMode.DUAL_INPUT

evaluate_dual_input_models_in_directory(MODELS_DIR, TRAIN_FOLDER, TEST_FOLDER, ALPHA, BETA, MODE)
