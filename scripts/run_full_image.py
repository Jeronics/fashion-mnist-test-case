import os.path as path
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.networks import ProductionCNN2
from utils.config import DATA_DIR, MEAN_PIXEL, STD_PIXEL, LIST_CLASS, ARTIFACTS_DIR
from utils.data_loaders import CustomFashionMNIST
from utils.application_utils import get_label_and_bounding_box, create_image


if __name__ == "__main__":
    np.random.seed(1)

    image_name = "test_long_image"
    pil_image = create_image(image_name)

    default_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
    ])

    pil_image = default_transformation(pil_image)
    pil_image = pil_image.unsqueeze(0)

    model_name = 'cnn_3'

    # loading
    with open(path.join(ARTIFACTS_DIR, model_name + '.pkl'), 'rb') as f:
        cvGridSearch = pickle.load(f)

    model = ProductionCNN2(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()

    print(type(pil_image))
    with torch.no_grad():
        class_idx, image_with_contour = get_label_and_bounding_box(model, pil_image)
