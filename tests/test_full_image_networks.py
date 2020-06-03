import pickle
from os import path
import torch
import torchvision.transforms as transforms
from models.skorch_networks import CustomNeuralNetClassifier
from utils.data_loaders import CustomFashionMNIST
from torch.utils.data import DataLoader
from utils.model_evaluation import ModelEvaluator
from models.networks import ProductionCNN, ProductionCNN2
from utils.config import ARTIFACTS_DIR, DATA_DIR, MEAN_PIXEL, STD_PIXEL


if __name__ == '__main__':
    model_name = 'cnn_2'

    default_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
    ])
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=default_transformation)

    # loading
    with open(path.join(ARTIFACTS_DIR, model_name + '.pkl'), 'rb') as f:
        cvGridSearch = pickle.load(f)

    model = ProductionCNN2(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()

    dataloader = DataLoader(testset, batch_size=100, shuffle=True)

    modelEvaluator = ModelEvaluator(None)
    modelEvaluator.y_real = []
    modelEvaluator.y_pred = []
    for i, (inputs, labels) in enumerate(dataloader):
        y_pred = torch.argmax(model.forward(inputs), dim=1)[:, 0, 0].tolist()
        modelEvaluator.y_real.extend(labels)
        modelEvaluator.y_pred.extend(y_pred)

    print(modelEvaluator.get_accuracy())
    print(modelEvaluator.get_confusion_matrix())
    modelEvaluator.save_results('full_image_' + model_name)
