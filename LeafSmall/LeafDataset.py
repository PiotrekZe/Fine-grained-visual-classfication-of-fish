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
        df = pd.read_csv(os.path.join(self.path, "all.csv")).iloc[:, 1:]

        for idx, row in df.iterrows():
            images.append(os.path.join(self.path, row['id']))
            targets.append(row['y'])

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42,
                                                            stratify=targets)

        return X_train, X_test, y_train, y_test
