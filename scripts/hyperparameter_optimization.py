import torch
import torchvision.transforms as transforms

from models.networks import CNN2
from models.skorch_networks import CustomNeuralNetClassifier
from utils.config import MEAN_PIXEL, STD_PIXEL, DATA_DIR
from utils.data_loaders import CustomFashionMNIST
from utils.model_evaluation import ModelEvaluator
from utils.model_selection import GridSearchCV

if __name__ == '__main__':
    ## Variables to change
    # Where you wish the model to be stored in the artifacts folder.
    model_name = "cnn_name"
    # Data augmentation on the training dataset or not.
    data_augmentation = True
    # CNN or CNN2 networks.
    Network = CNN2
    # Apply hyperparameter search or just train.
    hyperparameter_search = True
    # Dictionary with list of values per hyperparameter for grid search.
    params_options = {
        'lr': [0.001],
    }

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
    if data_augmentation:
        transformations = {
            "default": default_transformation,
            "train": {
                0: vertical_flip,
                1: vertical_flip,
                2: vertical_flip,
                3: vertical_flip,
                4: vertical_flip,
                5: default_transformation,
                6: vertical_flip,
                7: default_transformation,
                8: vertical_flip,
                9: default_transformation,
            }
        }
    else:
        transformations = {
            "default": default_transformation,
            "train": default_transformation,
        }

    trainset = CustomFashionMNIST(DATA_DIR, download=True, train=True, transform=transformations['train'])
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=transformations['default'])

    net = CustomNeuralNetClassifier(Network,
                                    max_epochs=1,
                                    lr=0.001,
                                    criterion=torch.nn.CrossEntropyLoss,
                                    optimizer__weight_decay=0.01,
                                    # Shuffle training data on each epoch
                                    iterator_train__shuffle=False,
                                    valid_transform=default_transformation,
                                    )

    cvGridSearch = GridSearchCV(net, params_options, cv=3, refit=True, name=model_name, transformation=transformations)

    if hyperparameter_search:
        cvGridSearch.fit(trainset)
    else:
        cvGridSearch.final_fit(trainset)

    y_pred = cvGridSearch.predict(testset)

    model_evaluator = ModelEvaluator(cvGridSearch.best_model)
    model_evaluator.fit(testset)
    print(model_evaluator.get_accuracy())
    print(model_evaluator.get_confusion_matrix())
    model_evaluator.show_training_accuracy_epoch()
    model_evaluator.show_training_loss_epoch()
