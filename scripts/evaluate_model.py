import pickle
from utils.model_evaluation import ModelEvaluator
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from utils.data_loaders import CustomFashionMNIST
from skorch.callbacks import BatchScoring
from skorch.dataset import CVSplit
from torch.utils.data import DataLoader
from os import path
from utils.config import ARTIFACTS_DIR, DATA_DIR, RESULTS_DIR
if __name__ == '__main__':

    model_name = 'cnn_3'
    # loading
    with open(path.join(ARTIFACTS_DIR, model_name+'.pkl'), 'rb') as f:
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
    modelEvaluator.save_results(cvGridSearch.name)
