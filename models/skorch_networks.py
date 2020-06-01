from copy import copy

import torch
import torchvision.transforms as transforms
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from torch.utils.data import Subset
from utils.config import MEAN_PIXEL, STD_PIXEL


class CustomNeuralNetClassifier(NeuralNetClassifier):

    def __init__(
            self,
            module,
            *args,
            valid_transform=None,
            criterion=torch.nn.CrossEntropyLoss,
            train_split=CVSplit(5, stratified=True),
            classes=None,
            **kwargs):
        super().__init__(module, *args, criterion=criterion, train_split=train_split,
                         classes=classes, **kwargs)

        if valid_transform is None:
            self.valid_transform = default_transformation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
            ])
        else:
            self.valid_transform = valid_transform

    def get_split_datasets(self, X, y=None, **fit_params):
        """Get internal train and validation datasets.

        The validation dataset can be None if ``self.train_split`` is
        set to None; then internal validation will be skipped.

        Override this if you want to change how the net splits
        incoming data into train and validation part.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        **fit_params : dict
          Additional parameters passed to the ``self.train_split``
          call.

        Returns
        -------
        dataset_train
          The initialized training dataset.

        dataset_valid
          The initialized validation dataset or None

        """
        dataset = self.get_dataset(X, y)
        if self.train_split:
            dataset_train, dataset_valid = self.train_split(
                dataset, y, **fit_params)
            if isinstance(dataset_valid, Subset):
                dataset_valid.dataset = copy(dataset_valid.dataset)
                dataset_valid.dataset.transform = self.valid_transform
            elif dataset_valid is not None:
                dataset_valid = copy(dataset_valid)
                dataset_valid.transform = self.valid_transform
        else:
            dataset_train, dataset_valid = dataset, None
        return dataset_train, dataset_valid
