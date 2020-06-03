############################# IMPORTS #############################

import cv2
import torch
import torchvision.transforms as transforms

from utils.application_utils import load_models, get_label_and_bounding_box
from utils.config import MEAN_PIXEL, STD_PIXEL, LIST_CLASS

###################################################################

if __name__ == '__main__':
    cv2.namedWindow("Classification App")
    vc = cv2.VideoCapture(0)

    count = 0
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    # Load models
    models = load_models(-1)
    idx = 0

    model = models[idx]
    bounding_box = None
    class_idx = None

    while rval:
        count += 1
        rval, frame = vc.read()
        cv2.imshow("Classification App", frame)

        key = cv2.waitKey(5)
        if key == 27:  # exit on ESC
            break
        elif key == 49:
            idx = (idx + 1) % len(models)
            print("model", idx)
            model = models[idx]

        camera = cv2.VideoCapture(0)
        _, frame = camera.read()
        if count > 5:
            with torch.no_grad():
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                default_transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((MEAN_PIXEL,), (STD_PIXEL,))
                ])
                A = default_transformation(frame)
                A = A.unsqueeze(0)
                class_idx, bounding_box = get_label_and_bounding_box(model, A, voting=True, display=False)
                print("Bounding Box\t", bounding_box, "Class:",
                      LIST_CLASS[class_idx] if class_idx is not None else None)
                count = 0
        if bounding_box is not None:
            cv2.rectangle(frame, bounding_box[:2],
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 255, 255), 2)

    vc.release()
    cv2.destroyWindow("Classification App")
