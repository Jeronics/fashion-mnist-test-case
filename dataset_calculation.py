import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np

def getStatistics():
    '''
    Calculates the MEAN and STD for a given dataset.
    :param df: A pandas dataframe
    :return:
    '''

    # Define the transformations applied to the training data
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    trainset = FashionMNIST('../data/', download=True, train=False, transform=transform)

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    means = list()
    variances = list()
    for i, (img, label) in enumerate(train_loader):
        means.extend([np.mean(image.numpy()) for image in img])
        variances.extend([np.std(image.numpy()) for image in img])
    print(np.mean(variances))
    return np.mean(means), np.std(means)



if __name__ == '__main__':
    mean, std = getStatistics()
    print(mean, std)
