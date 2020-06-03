import pickle
import os.path as path

import torchvision.transforms as transforms

from utils.config import ARTIFACTS_DIR, DATA_DIR, MEAN_PIXEL, STD_PIXEL
from utils.data_loaders import CustomFashionMNIST
from utils.model_evaluation import ModelEvaluator

if __name__ == '__main__':
    # Variable to change.
    model_name = 'cnn_3'

    # load saved pkl grid search model
    with open(path.join(ARTIFACTS_DIR, model_name + '.pkl'), 'rb') as f:
        cvGridSearch = pickle.load(f)

    # Define the transformations to normalize the data.
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))]
    )

    # Instantiate dataset
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=transform)

    # Fit ModelEvaluator and save results in the results folder under the model_name+'pkl'
    modelEvaluator = ModelEvaluator(cvGridSearch.best_model)
    modelEvaluator.fit(testset)
    modelEvaluator.save_results(model_name)
    print(modelEvaluator.get_accuracy())
    print(modelEvaluator.get_confusion_matrix())
