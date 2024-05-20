from src.data.data_processing import InteractionMapMode
from evaluate_models import evaluate_models_in_directory

TRAIN_FOLDER = '../data/training_data/'
TEST_FOLDER = '../data/true_set/'
MODELS_DIR = '../models/beta'
CHAINS = ['TRB_CDR3']
MODE = InteractionMapMode.SINGLE

evaluate_models_in_directory(MODELS_DIR, TRAIN_FOLDER, TEST_FOLDER, CHAINS, MODE)
