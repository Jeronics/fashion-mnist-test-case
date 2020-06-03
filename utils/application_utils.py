############################# IMPORTS #############################

import os.path as path
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.networks import ProductionCNN, ProductionCNN2
from utils.config import ARTIFACTS_DIR, DATA_DIR
from utils.data_loaders import CustomFashionMNIST

###################################################################


def create_image(image_name):
    '''
    Creates an image by joining various
    :param image_name: The name where the image will be saved
    :return: a large combined image.
    '''
    chosen_class = 2
    # Read ImageNet class id to name mapping
    null_transformation = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=null_transformation)

    full_image2 = []
    once = True
    for i in range(3):
        full_image = []
        shuffled_classes = list(range(0, 20))
        np.random.shuffle(shuffled_classes)
        for c in shuffled_classes:
            mask = np.where(dataset.targets == c % 10)[0]
            rand_idx = np.random.randint(len(mask))
            idx = mask[rand_idx]
            img = dataset[idx][0][0, :, :] * (1 if once and c == chosen_class and i == 1 else 0)
            if once and c == 9 and i == 1:
                once = False
            full_image.append(img.numpy())
        horiz = np.hstack(np.array(full_image))
        full_image2.append(horiz)
    imgs_comb = np.vstack(np.array(full_image2)) * 255

    imgs_comb = Image.fromarray(imgs_comb)
    if imgs_comb.mode != 'RGB':
        imgs_comb_save = imgs_comb.convert('RGB')

    save_dir = path.join(DATA_DIR, 'full_images')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    imgs_comb_save.save(path.join(save_dir, image_name + '.jpg'))
    return imgs_comb


def get_maximum_class(prediction_values, predicted_classes):
    '''
    Get class with the highest predicted value.
    :param prediction_values: List of predicted values
    :param predicted_classes: List of predicted classes
    :return: The predicted class.
    '''
    row_max, row_idx = torch.max(prediction_values, dim=1)
    col_max, col_idx = torch.max(row_max, dim=1)
    predicted_class = predicted_classes[0, row_idx[0, col_idx], col_idx]
    return predicted_class


def get_most_voted_class(prediction_values, predicted_classes):
    '''
    Get most voted class with high predicted values
    :param prediction_values: List of prediction values
    :param predicted_classes: List of predicted classes
    :return: predicted class
    '''
    # Apply voting to each class
    classes, counts = np.unique(predicted_classes[prediction_values > np.percentile(prediction_values, 90)],
                                return_counts=True)
    if len(counts) == 0:
        return get_maximum_class(prediction_values, predicted_classes)
    predicted_class = torch.tensor([classes[np.argmax(counts)]])
    return predicted_class


def get_label_and_bounding_box(model, image, voting=True, display=False):
    '''
    Returns the predicted label and bounding box.
    :param model: a nn.Model which accepts images of any size (at least larger than 28x28)
    :param image: a 28x28 grayscale image or larger.
    :param voting: whether to use the voting method to determine the predicted class or the maximum method.
    :param display: whether to display the image.
    :return: the predicted class and the bounding box.
    '''
    predictions = model(image)
    predictions = torch.softmax(predictions, dim=1)

    # Find the class with the maximum score in the n x m output map
    prediction_values, predicted_classes = torch.max(predictions, dim=1)

    if voting:
        # Apply voting to each class
        predicted_class = get_most_voted_class(prediction_values, predicted_classes)
    else:
        # Get class with the maximum value
        predicted_class = get_maximum_class(get_maximum_class(prediction_values, predicted_classes))

    # Find the n x m score map for the predicted class
    score_map = predictions[0, predicted_class, :, :].cpu().numpy()
    score_map = score_map[0]

    # Resize score map to the original image size
    score_map = cv2.resize(score_map, (image.shape[3], image.shape[2]))

    # Binarize score map
    _, score_map_for_contours = cv2.threshold(score_map, np.percentile(score_map, 90), 1.0, type=cv2.THRESH_BINARY)
    score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()

    # Find the countour of the binary blob
    contours, _ = cv2.findContours(score_map_for_contours, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Find bounding box around the object.
    if len(contours) == 0:
        return None, None
    bounding_box = cv2.boundingRect(contours[0])

    return predicted_class, bounding_box


def load_models(load=-1):
    '''
    Loads one or various models
    :param load: a variable indicatin which model to load (1, 2, 3) or all (-1)
    :return: a list of models
    '''
    models = []
    # loading
    if load == 1 or load == -1:
        model_1_name = 'cnn_1'
        with open(path.join(ARTIFACTS_DIR, model_1_name + '.pkl'), 'rb') as f:
            cvGridSearch = pickle.load(f)
        generator_A = ProductionCNN(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()
        models.append(generator_A)
    if load == 2 or load == -1:
        model_2_name = 'cnn_2'
        # loading
        with open(path.join(ARTIFACTS_DIR, model_2_name + '.pkl'), 'rb') as f:
            cvGridSearch = pickle.load(f)

        generator_B = ProductionCNN2(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()
        models.append(generator_B)
    if load == 3 or load == -1:
        model_3_name = 'cnn_3'
        # loading
        with open(path.join(ARTIFACTS_DIR, model_3_name + '.pkl'), 'rb') as f:
            cvGridSearch = pickle.load(f)

        generator_C = ProductionCNN2(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()
        models.append(generator_C)
    return models
