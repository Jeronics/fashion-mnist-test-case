import os
from os import path
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from utils.config import RESULTS_DIR


class ModelEvaluator:

    def __init__(self, model):
        self.model = model
        self.y_real = None
        self.y_pred = None

    def fit(self, testset):
        if isinstance(testset, torch.utils.data.Subset):
            self.y_real = testset.dataset.targets[testset.indices]
        else:
            self.y_real = testset.targets
        self.y_pred = self.model.predict(testset)

    def get_accuracy(self):
        if self.y_pred is not None:
            return accuracy_score(self.y_real, self.y_pred)
        return None

    def get_confusion_matrix(self):
        if self.y_pred is not None:
            return confusion_matrix(self.y_real, self.y_pred)
        return None

    def display_confusion_matrix(self, save_plot=None):
        if self.y_pred is not None:
            print(self.get_confusion_matrix())

    def show_training_accuracy_epoch(self, figname=None, show=True):
        train_acc = self.model.history[:, 'train_acc'][1:]
        valid_acc = self.model.history[:, 'valid_acc'][:-1]
        plt.figure()
        plt.axhline(0.95, ls=":", c="black", label="State of the Art")
        if self.y_pred is not None:
            plt.axhline(self.get_accuracy(), ls="-.", c='g', label="Test Accuracy")
        plt.plot(range(1, len(train_acc) + 1), train_acc, 'b', label='Training Accuracy')
        plt.plot(range(1, len(valid_acc) + 1), valid_acc, 'r', label='Validation Accuracy')
        plt.legend()
        plt.ylim([0.0, 1.0])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig('Train-Val-loss-1.png')
        if show:
            plt.show()
        if figname is not None:
            plt.savefig(figname)

    def show_training_loss_epoch(self, figname=None, show=True):
        train_loss = self.model.history[:-1, 'train_loss'][1:]
        valid_loss = self.model.history[:, 'valid_loss'][:-1]
        plt.figure()
        plt.plot(range(1, len(train_loss) + 1), train_loss, 'b', label='Training Loss')
        plt.plot(range(1, len(valid_loss) + 1), valid_loss, 'r', label='Validation Loss')
        plt.ylim([0.0, 1.0])
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('Train-Val-loss-1.png')
        if show:
            plt.show()
        if figname is not None:
            plt.savefig(figname)

    def save_results(self, name):
        save_dir = path.join(RESULTS_DIR, name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print(path.exists(save_dir))
        if self.y_pred is not None:
            self.show_training_accuracy_epoch(figname=os.path.join(save_dir, "accuracy_epoch.png"), show=False)
            self.show_training_loss_epoch(figname=os.path.join(save_dir, "loss_epoch.png"), show=False)
            np.savetxt("foo.csv", self.get_confusion_matrix(), delimiter=",")
        else:
            print(f"Could not save results. Please train the {self.__name__} first")