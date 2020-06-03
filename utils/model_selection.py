import itertools
import pickle
from copy import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skorch.callbacks import BatchScoring
from skorch.callbacks import Callback
from skorch.dataset import CVSplit
from torch.utils.data import Subset
from utils.config import ARTIFACTS_DIR
from os import path
from utils.model_evaluation import ModelEvaluator

class GridSearchCV:
    '''
    Although I could have used See this issue: https://github.com/skorch-dev/skorch/issues/443
    '''

    def __init__(self, model, parameter_grid, cv=3, refit=True, name="my_model", transformation=None):
        self.model = model
        self.parameter_grid = parameter_grid
        self.refit = refit
        self.cv = cv
        self.best_model = model
        self.name = name
        self.transformation = transformation

        # Initialized parameters that will be used further on
        self.best_params_ = None
        self.best_model_score_ = None
        self.best_model_score_train = None

    def fit(self, dataset):
        list_params = self.get_parameter_combination(self.parameter_grid)
        param_valid_scores = []
        param_train_scores = []
        dataset_copy = copy(dataset)
        if self.transformation is not None:
            dataset_copy.transform = self.transformation['default']

        for k in range(self.cv):
            train_idxs, val_idxs = self.split_list_fold(len(dataset), cv=self.cv, fold=k)
            train, val = Subset(dataset, train_idxs), Subset(dataset_copy, val_idxs)

            # For each Parameter
            valid_scores_per_fold = []
            train_scores_per_fold = []
            for i, param in enumerate(list_params):
                # set current parameters
                self.model.set_params(**param)

                # fit on the training set
                self.model.fit(train, None)

                model_evaluator = ModelEvaluator(self.model)

                # evaluate on the validation set
                model_evaluator.fit(train)
                train_scores_per_fold.append(model_evaluator.get_accuracy())

                # evaluate on the validation set
                model_evaluator.fit(val)
                valid_scores_per_fold.append(model_evaluator.get_accuracy())

            param_valid_scores.append(valid_scores_per_fold)
            param_train_scores.append(train_scores_per_fold)

        # Find the best parameter combination:
        mean_valid_scores = np.mean(param_valid_scores, axis=0)
        mean_train_scores = np.mean(param_train_scores, axis=0)
        self.best_model_score_ = np.max(mean_valid_scores)
        self.best_model_score_train = np.max(mean_train_scores)
        self.best_params_ = list_params[np.argmax(mean_valid_scores)]
        best_score = np.max(mean_valid_scores)
        print("Best params:\t" + str(self.best_params_))
        print("Best score:\t" + str(best_score))
        if self.refit:
            self.final_fit(dataset, train_all=False, best_params=self.best_params_)

    def predict(self, dataset):
        return self.model.predict(dataset)

    def final_fit(self, dataset, best_params=None, train_all=False):
        if best_params is None:
            best_params = self.model.get_params()
        # Once the best parameters are found, the best model is trained on the whole training set.
        if not self.model.train_split is None and isinstance(self.model.train_split, CVSplit):
            best_params.update({"train_split": None if train_all else CVSplit(5)})
        # Callbacks
        train_acc = BatchScoring(scoring='accuracy', on_train=True,
                                 name='train_acc', lower_is_better=False)
        # Callbacks
        valid_acc = BatchScoring(scoring='accuracy', on_train=False,
                                 name='valid_acc', lower_is_better=False)
        best_params.update({"callbacks": [train_acc, valid_acc]})

        self.best_model.set_params(**best_params)
        self.best_model.fit(dataset, None)
        # saving
        with open(path.join(ARTIFACTS_DIR, self.name + '.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def get_parameter_combination(self, params):
        '''
        Returns a list of possible combinations of parameters.
        # Credit to https://codereview.stackexchange.com/questions/171173/list-all-possible-permutations-from-a-python-dictionary-of-lists
        :param params: A dictionary of parameters with list values.
        :return: A list of dictionaries
        '''
        keys, values = zip(*params.items())
        list_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return list_params

    def split_list_fold(self, list_size, cv=3, fold=0):
        '''
        Splits a list for each cross validation step.

        :param list_size: lengths of a list
        :param cv: the total number of total folds to iterate over.
        :param fold: A fold.
        :return: returns a list of the remaining and selected fold
        '''
        left_size = int(list_size / cv * fold)
        right_size = int(list_size / cv * (fold + 1))

        # Find split point
        indexes = list(range(list_size))

        # Get index list
        fold_list = indexes[left_size: right_size]
        remaining_list = indexes[:left_size] + indexes[right_size:]
        return remaining_list, fold_list


class WeightTracker(Callback):

    def __init__(self):
        super(Callback, self).__init__()

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        print("epoch begin",id(dataset_train), id(dataset_valid))

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        print("epoch begin",id(dataset_train),id(dataset_valid))

    def on_grad_computed(self, net, named_parameters,
                         X=None, y=None, training=None, **kwargs):
        print(torch.norm(net.module_.conv1.weight.grad).data.item())

