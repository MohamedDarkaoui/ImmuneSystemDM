import matplotlib.pyplot as plt


def plot_loss(history, filename=None):
    epochs = range(1, len(history.history['loss']) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history.history['loss'], color='blue', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], color='green', label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_metrics(history, filename=None):
    epochs = range(1, len(history.history['accuracy']) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history.history['accuracy'], 'b', label='Accuracy', color='blue')
    if 'val_accuracy' in history.history:
        plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation Accuracy', color='cyan',
                 linestyle='dashed')

    plt.plot(epochs, history.history['roc_auc'], 'b', label='ROC AUC', color='red')
    if 'val_roc_auc' in history.history:
        plt.plot(epochs, history.history['val_roc_auc'], 'b', label='Validation ROC AUC', color='pink',
                 linestyle='dashed')

    plt.title('Training Metrics Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Values')

    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_auc(auc_scores, filename=None):
    auc_scores = dict(sorted(auc_scores.items(), key=lambda item: item[1], reverse=True))
    average_auc_score = sum(auc_scores.values()) / len(auc_scores)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(auc_scores.keys(), auc_scores.values(), color='skyblue')
    plt.xlabel('Epitopes')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores for Different Epitopes')
    plt.ylim(0, 1)
    plt.xticks(rotation='vertical')

    # vertical line for the average AUC score
    plt.axhline(y=average_auc_score, color='red', linestyle='--', linewidth=1)
    plt.text(x=len(auc_scores) - 1, y=average_auc_score, s='Average', color='red', va='bottom')

    # indicate the 50%
    plt.axhline(y=0.5, color='purple', linestyle='--', linewidth=1)
    plt.text(x=len(auc_scores) - 1, y=0.5, s='0.5', color='red', va='bottom')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()