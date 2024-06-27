import matplotlib.pyplot as plt
import pandas as pd
from transformers import TrainerCallback

def plot_metrics(logs, filename='training_process.png'):
    # Convert logs to DataFrame for easier plotting
    df = pd.DataFrame(logs)
    plt.figure(figsize=(16, 8))

    # Plot loss
    plt.subplot(1, 3, 1)
    if 'loss' in df:
        plt.plot(df['epoch'], df['loss'], label='Training Loss')
    else:
        print("Warning: 'loss' key not found in logs.")
    if 'eval_loss' in df:
        plt.plot(df['epoch'], df['eval_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    if 'accuracy' in df:
        plt.plot(df['epoch'], df['accuracy'], label='Training Accuracy')
    if 'eval_accuracy' in df:
        plt.plot(df['epoch'], df['eval_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot learning rate
    plt.subplot(1, 3, 3)
    if 'learning_rate' in df:
        plt.plot(df['epoch'], df['learning_rate'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.legend()

    # Save the figure
    plt.savefig(filename)

class CustomCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.logs = []

    def on_log(self, args, state, control, **kwargs):
        self.logs.append(kwargs['logs'])

    def get_logs(self):
        return self.logs
