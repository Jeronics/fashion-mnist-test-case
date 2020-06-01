import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url

from PIL import Image
import cv2
import pickle
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loaders import CustomFashionMNIST
from utils.config import DATA_DIR, MEAN_PIXEL, STD_PIXEL, LIST_CLASS, ARTIFACTS_DIR
import torchvision.transforms as transforms
from models.networks import ProductionCNN

def join_images(dataset, image_name):
    full_image2 = []
    once = True
    for i in range(3):
        full_image = []
        shuffled_classes = list(range(0, 20))
        np.random.shuffle(shuffled_classes)
        for c in shuffled_classes:
            mask = np.where(dataset.targets==c%10)[0]
            rand_idx = np.random.randint(len(mask))
            idx = mask[rand_idx]
            img = dataset[idx][0][0,:,:]*(1 if once and c==9 and i==0 else 0)
            if once and c == 9 and i == 2:
                once=False
            full_image.append(img.numpy())
        horiz = np.hstack(np.array(full_image))
        full_image2.append(horiz)
    imgs_comb = np.vstack(np.array(full_image2))*255

    imgs_comb = Image.fromarray(imgs_comb)
    if imgs_comb.mode != 'RGB':
        imgs_comb_save = imgs_comb.convert('RGB')

    imgs_comb_save.save('Trifecta_vertical.jpg')
    return imgs_comb


if __name__ == "__main__":
    # Read ImageNet class id to name mapping
    null_transformation = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = CustomFashionMNIST(DATA_DIR, download=True, train=False, transform=null_transformation)
    image_name = "test_long_image"
    pil_image = join_images(testset,image_name)

    default_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
    ])

    pil_image = default_transformation(pil_image)
    pil_image = pil_image.unsqueeze(0)

    model_name = 'cnn_1'

    # loading
    with open(path.join(ARTIFACTS_DIR, model_name + '.pkl'), 'rb') as f:
        cvGridSearch = pickle.load(f)

    model = ProductionCNN(state_dict=cvGridSearch.best_model.module_.state_dict()).eval()


    with torch.no_grad():
        # Perform inference.
        # Instead of a 1x1000 vector, we will get a
        # 1x1000xnxm output ( i.e. a probabibility map
        # of size n x m for each 1000 class,
        # where n and m depend on the size of the image.)
        preds = model(pil_image)
        preds = torch.softmax(preds, dim=1)
        print(preds.shape)

        print('Response map shape : ', preds.shape)

        # Find the class with the maximum score in the n x m output map
        pred, class_idx = torch.max(preds, dim=1)
        print(pred.shape)
        row_max, row_idx = torch.max(pred, dim=1)
        col_max, col_idx = torch.max(row_max, dim=1)
        predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]

        # Print top predicted class
        print('Predicted Class : ', LIST_CLASS[predicted_class], predicted_class)

        # Find the n x m score map for the predicted class
        score_map = preds[0, predicted_class, :, :].cpu().numpy()
        score_map = score_map[0]

        # Resize score map to the original image size
        score_map = cv2.resize(score_map, (pil_image.shape[1], pil_image.shape[0]))

        print("Score map")
        print(score_map)
        # Binarize score map
        _, score_map_for_contours = cv2.threshold(score_map, 0.01, 1, type=cv2.THRESH_BINARY)
        print(score_map_for_contours)
        score_map_for_contours = score_map_for_contours.astype(np.uint8).copy()

        print(score_map_for_contours)
        # Find the countour of the binary blob
        contours, _ = cv2.findContours(score_map_for_contours, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # Find bounding box around the object.
        rect = cv2.boundingRect(contours[0])

        print("HEeu")
        print(score_map)
        # Apply score map as a mask to original image
        score_map = score_map - np.min(score_map[:])
        score_map = score_map / np.max(score_map[:])

        print(score_map)
        print(pil_image * torch.from_numpy(score_map))

        score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
        print("YO")
        print(score_map)
        masked_image = (pil_image * score_map).astype(np.uint8)

        # Display bounding box
        cv2.rectangle(masked_image, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)


        # Display images
        cv2.imshow("Original Image", pil_image)
        cv2.imshow("scaled_score_map", score_map)
        cv2.imshow("activations_and_bbox", masked_image)
        cv2.waitKey(0)