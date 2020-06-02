import pickle
from os import path
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from models.skorch_networks import CustomNeuralNetClassifier
from utils.data_loaders import CustomFashionMNIST
from torch.utils.data import DataLoader
from utils.model_evaluation import ModelEvaluator
from models.networks import ProductionCNN
from utils.config import ARTIFACTS_DIR, DATA_DIR, MEAN_PIXEL, STD_PIXEL

if __name__ == '__main__':
    model_name = 'cnn_1'

    default_transformation = Compose([
        ToTensor(),
        Normalize((MEAN_PIXEL,), (STD_PIXEL,))
    ])
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=default_transformation)

    # loading
    with open(path.join(ARTIFACTS_DIR, model_name + '.pkl'), 'rb') as f:
        cvGridSearch = pickle.load(f)

    model = ProductionCNN(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()

    # net = CustomNeuralNetClassifier(model,
    #                                 max_epochs=1000,
    #                                 lr=0.001,
    #                                 criterion=torch.nn.CrossEntropyLoss,
    #                                 optimizer__weight_decay=0.01,
    #                                 # Shuffle training data on each epoch
    #                                 iterator_train__shuffle=True,
    #                                 valid_transform=default_transformation,
    #                                 )

    dataloader = DataLoader(testset, batch_size=100)

    modelEvaluator = ModelEvaluator(None)
    modelEvaluator.y_real=[]
    modelEvaluator.y_pred=[]
    for i, (inputs, labels) in enumerate(dataloader):
        y_pred = torch.argmax(model.forward(inputs), dim=1)[:,0,0].tolist()
        modelEvaluator.y_real.extend(labels)
        modelEvaluator.y_pred.extend(y_pred)

    print(modelEvaluator.get_accuracy())
    print(modelEvaluator.get_confusion_matrix())
    # modelEvaluator.save_results(model_name)
    # print(cvGridSearch.best_params_)
    # print(cvGridSearch.best_model.module_)
