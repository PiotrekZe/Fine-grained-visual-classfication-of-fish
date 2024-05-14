import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random
import pandas


class Dataset:
    def __init__(self, path):
        self.path = path

    def read_dataset(self):
        images, targets = [], []
        folders = os.listdir(self.path)

        for idx, folder in enumerate(folders):
            folder_path = os.path.join(self.path, folder)
            folder_path = os.path.join(folder_path, "valid")
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                images.append(file_path)
                targets.append(idx)

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42,
                                                            stratify=targets)

        return X_train, X_test, y_train, y_test
