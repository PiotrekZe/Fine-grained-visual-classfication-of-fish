import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random
import pandas


class Dataset:
    def __init__(self, path, size):
        self.path = path
        self.size = size

    def read_dataset(self):
        X_train, y_train, X_test, y_test = [], [], [], []

        folders = os.listdir(os.path.join(self.path, "Training_Set"))
        for idx, folder in enumerate(folders):
            folder_path = os.path.join(os.path.join(self.path, "Training_Set"), folder)
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                image = Image.open(file_path).convert("RGB")
                image = image.resize((self.size, self.size), resample=Image.BILINEAR)
                X_train.append(np.array(image))
                y_train.append(idx)

        folders = os.listdir(os.path.join(self.path, "Test_Set"))
        for idx, folder in enumerate(folders):
            folder_path = os.path.join(os.path.join(self.path, "Test_Set"), folder)
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                image = Image.open(file_path).convert("RGB")
                image = image.resize((self.size, self.size), resample=Image.BILINEAR)
                X_test.append(np.array(image))
                y_test.append(idx)

        return X_train, X_test, y_train, y_test
