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
        print(idx, len(folders))
        # images = images[:1000]
        # targets = targets[:1000]
        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42,
                                                            stratify=targets)

        return X_train, X_test, y_train, y_test

    def read_dataset_order(self):
        images, targets = [], []
        folders = os.listdir(self.path)

        data = pd.read_csv("E:/AI/FGVC_ICIST/train.csv")
        dict_order = dict()
        for idx, row in data.iterrows():
            if row["Folder"] not in dict_order:
                dict_order[row["Folder"]] = row['Order']
        order_el = list(set(dict_order.values()))
        class_idx = dict(zip(order_el, list(range(len(order_el)))))

        for folder in folders:
            if folder in dict_order.keys():
                folder_path = os.path.join(self.path, folder)
                files = os.listdir(folder_path)
                for file in files:
                    if file.endswith(".jpg") or file.endswith(".png"):
                        file_path = os.path.join(folder_path, file)
                        images.append(file_path)
                        targets.append(class_idx[dict_order[folder]])

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42,
                                                            stratify=targets)

        return X_train, X_test, y_train, y_test
