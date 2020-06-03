# Advertima test case
FashionMNIST classification and object detection.

## 0. Setup

This program runs on python 3.7. To install all dependencies, run the following in the console inside the directory of the project:

```pip install -r requirements.txt```

## 1. Report

The report can be found in the ```notebooks/test_case_report.ipynb``` notebook, which contains a full report about the project, the approaches, the result and the app.

## 2. Training and Hyperparameter tuning 

To train or tune a model, run the ```hyperparameter_optimization``` script.
Before you do so, tweak the following variables in the script:

```
## Variables to change
# Where you wish the model to be stored in the artifacts folder.
model_name = "cnn_1_500"
# Data augmentation on the training dataset or not.
data_augmentation = False
# CNN or CNN2 networks.
Network = CNN
# Apply hyperparameter search or just train.
hyperparameter_search = False
# Dictionary with list of values per hyperparaeter for grid search.
params_options = {   
    'lr': [0.001],
}
```

and run:

```python -m scripts.hyperparameter_optimization```

This script will save the model as a pickle in the artifacts folder.

## 3. Evaluation

Once the model has been trained, The ```evaluate_model.py``` script will be used to extract results of the classifier such as plots and metrics.

In order to run, change the model_name to fit with one of the existing pkl. file names

```python -m scripts.evaluate_model```

## 4. Application

Run the application consists in applying the previous object detection to a live feed. To do this, I used openCV's library to capture the images. Unfortunately, this part was not completely finished because of the lack of time.
To run the code:

```python -m scripts.cam_application```

Once running,
- Press Esc to stop the application.
- Press 1 to change the model.

