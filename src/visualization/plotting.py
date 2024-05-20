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
    epochs = range(1, len(history.history['roc_auc']) + 1)

    plt.figure(figsize=(8, 6))

    plt.plot(epochs, history.history['roc_auc'], 'b', label='ROC AUC')
    if 'val_roc_auc' in history.history:
        plt.plot(epochs, history.history['val_roc_auc'], 'b', label='Validation ROC AUC',
                 linestyle='dashed')

    plt.title('Training Metrics Across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Values')

    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_dict(data,  xlabel='X-axis', ylabel='Y-axis', title='Plot', average_line=True, half_line=True,
              filename=None):
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    average_value = sum(data.values()) / len(data)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.keys(), data.values(), color='skyblue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation='vertical')

    if average_line:
        plt.axhline(y=average_value, color='red', linestyle='--', linewidth=1)
        plt.text(len(data) - 1, average_value, f'Average: {average_value:.2f}', color='red', va='bottom', ha='right')

    if half_line:
        plt.axhline(y=0.5, color='purple', linestyle='--', linewidth=1)
        plt.text(len(data) - 1, 0.5, '0.5', color='purple', va='bottom', ha='right')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()