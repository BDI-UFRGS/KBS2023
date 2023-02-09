import matplotlib.pyplot as plt
import numpy as np

class Accuracy_Loss():
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    name = None

    def __init__(self, train_acc, train_loss, val_acc, val_loss, name) -> None:
        self.train_acc = train_acc
        self.train_loss = train_loss
        self.val_acc = val_acc
        self.val_loss = val_loss
        self.name = name
        
    def save_fig(self) -> None:
        epochs = list(range(1, len(self.val_acc) + 1))

        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        ax1 = plt.subplot(2, 1, 1)
        ax1.set_ylim((0, np.max(self.train_loss + self.val_loss)))
        # r is for "solid red line"
        plt.plot(epochs, self.train_loss, 'r', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, self.val_loss, 'b', label='Validation loss')
        plt.title(f'{self.name} - Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        ax2 = plt.subplot(2, 1, 2)
        ax2.set_ylim((0, 1))

        plt.plot(epochs, self.train_acc, 'r', label='Training acc')
        plt.plot(epochs, self.val_acc, 'b', label='Validation acc')
        plt.title(f'{self.name} - Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.savefig(f'{self.name}.png')
        plt.close()