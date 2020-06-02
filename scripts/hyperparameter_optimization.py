import torch
import torchvision.transforms as transforms
from skorch.helper import predefined_split
from torch.utils.data import random_split

import sys
from models.networks import CNN2
import  torchvision.transforms as transforms
from models.skorch_networks import CustomNeuralNetClassifier
from utils.config import MEAN_PIXEL, STD_PIXEL, DATA_DIR
from utils.data_loaders import CustomFashionMNIST
from utils.model_evaluation import ModelEvaluator
from utils.model_selection import GridSearchCV

if __name__ == '__main__':
    torch.manual_seed(0)
    # Define the transformations applied to the training data
    vertical_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
    ])

    default_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
    ])

    # transformations = {
    #     "default": default_transformation,
    #     "train": {
    #         0: vertical_flip,
    #         1: vertical_flip,
    #         2: vertical_flip,
    #         3: vertical_flip,
    #         4: vertical_flip,
    #         5: default_transformation,
    #         6: vertical_flip,
    #         7: default_transformation,
    #         8: vertical_flip,
    #         9: default_transformation,
    #     }
    # }
    transformations = {
        "default": default_transformation,
        "train": default_transformation,
    }

    trainset = CustomFashionMNIST(DATA_DIR, download=True, train=True, transform=transformations['train'])
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=transformations['default'])

    params_options = {
        'lr': [0.001],
    }

    net = CustomNeuralNetClassifier(CNN2,
                                    max_epochs=1000,
                                    lr=0.001,
                                    criterion=torch.nn.CrossEntropyLoss,
                                    optimizer__weight_decay=0.01,
                                    # Shuffle training data on each epoch
                                    iterator_train__shuffle=False,
                                    valid_transform=default_transformation,
                                    )

    cvGridSearch = GridSearchCV(net, params_options, cv=3, refit=True, name="cnn_2_1000", transformation=transformations)

    cvGridSearch.final_fit(trainset)

    y_pred = cvGridSearch.predict(testset)

    model_evaluator = ModelEvaluator(cvGridSearch.best_model)
    model_evaluator.fit(testset)
    print(model_evaluator.get_accuracy())
    print(model_evaluator.get_confusion_matrix())
    model_evaluator.show_training_accuracy_epoch()
    model_evaluator.show_training_loss_epoch()
