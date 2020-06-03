############################# IMPORTS #############################

import os.path as path
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms

from models.networks import ProductionCNN2
from utils.application_utils import get_label_and_bounding_box, create_image
from utils.config import MEAN_PIXEL, STD_PIXEL, ARTIFACTS_DIR

###################################################################


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
