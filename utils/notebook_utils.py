import numpy as np
import matplotlib.pyplot as plt
from utils.config import LIST_CLASS


def visualize_dataset_sample(dataset):
    fig = plt.figure(figsize=(12, 3))
    for c in range(0,10):
        mask = np.where(dataset.targets==c)[0]
        rand_idx = np.random.randint(len(mask))
        idx = mask[rand_idx]
        img = dataset[idx][0][0,:,:]
        fig.add_subplot(1, 10, c+1)
        plt.title(LIST_CLASS[c])
        plt.axis('off')
        plt.imshow(img, cmap='gray')





def get_output_shape(width, height, channels, kernel, stride=1, padding=0):
    out_width = ((width-kernel+2*padding)/stride)+1
    out_height = ((height-kernel+2*padding)/stride)+1
    return int(out_width), int(out_height), channels
