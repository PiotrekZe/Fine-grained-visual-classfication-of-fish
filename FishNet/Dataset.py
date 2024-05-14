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
        idx = 0
        for folder in folders:
            folder_path = os.path.join(self.path, folder)
            files = os.listdir(folder_path)
            if len(files) > 2:
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        file_path = os.path.join(folder_path, file)
                        images.append(file_path)
                        targets.append(idx)
            idx += 1

        # images = images[:1000]
        # targets = targets[:1000]
        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42,
                                                            stratify=targets)

        return X_train, X_test, y_train, y_test
