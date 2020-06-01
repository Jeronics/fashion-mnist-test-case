import pickle
from os import path

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.config import ARTIFACTS_DIR, DATA_DIR
from utils.data_loaders import CustomFashionMNIST
from utils.model_evaluation import ModelEvaluator

if __name__ == '__main__':
    model_name = 'cnn_1'

    # loading
    with open(path.join(ARTIFACTS_DIR, model_name + '.pkl'), 'rb') as f:
        cvGridSearch = pickle.load(f)

    # Define the transformations applied to the training data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=transform)

    train_loader = DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

    modelEvaluator = ModelEvaluator(cvGridSearch.best_model)
    modelEvaluator.fit(testset)
    print(modelEvaluator.get_accuracy())
    print(modelEvaluator.get_confusion_matrix())
    modelEvaluator.save_results(model_name)
    print(cvGridSearch.best_params_)
    print(cvGridSearch.best_model.module_)
