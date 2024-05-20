import os
import numpy as np
from src.data.data_processing import generate_imap_dataset, generate_test_data, generate_ranking_test_data, \
    InteractionMapMode
from src.models.TCR3DModel import TCR3DModel
from src.visualization.plotting import  plot_dict

def evaluate_models_in_directory(models_dir, train_folder, test_folder, chains, mode):
    df, shape = generate_imap_dataset(train_folder, chains, mode)

    test_data = generate_test_data(test_folder=test_folder, tcr_chains=chains, mode=mode, imap_shape=shape)

    all_auc_scores = []
    all_macro_auc = []
    all_micro_auc = []

    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    print(f"Found {len(model_dirs)} model directories in {models_dir}")

    for model_dir in model_dirs:
        model_path = os.path.join(models_dir, model_dir)

        print(f"Evaluating model in directory: {model_dir}")

        model = TCR3DModel(
            save_path=model_path,
            input_shape=shape,
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
        plot_dict(average_per_epitope_auc, 'Epitope', 'Average AUC',
                  'Average AUC for Different Epitopes', average_line=True)
    else:
        print("No models were successfully evaluated.")

    ranking_data = generate_ranking_test_data(test_folder, chains, mode, shape)

    if model:
        ranks = model.evaluate_rank(ranking_data)
        print("Ranks:", ranks)
        vals = ranks.values()
        print("Average Rank:", sum(vals) / len(vals))
    else:
        print("No valid model available for ranking evaluation.")