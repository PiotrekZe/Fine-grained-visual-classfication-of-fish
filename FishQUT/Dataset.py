import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import random
import pandas


class Dataset:
    def __init__(self, path, size, prob):
        self.path = path
        self.size = size
        self.prob = prob

    def read_dataset(self):
        images, targets = [], []
        p = "D:/Databases/20/fish_QUT/Fish_Data/final_all_index.txt"
        df = pd.read_csv(p, header=None, sep="=").iloc[:, [0, 4]]

        for idx, row in df.iterrows():
            file_path = os.path.join(self.path, f"{row[4]}.png")
            image = Image.open(file_path).convert("RGB")
            image = image.resize((self.size, self.size), resample=Image.BILINEAR)
            images.append(np.array(image))
            targets.append(row[0] - 1)

        X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42,
                                                            stratify=targets)

        if self.prob != 0.0:
            X_train, y_train = self.__augment_data(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def __horizontal_flip(self, images, labels):
        horizontal_images, horizontal_labels = [], []
        for i in range(len(images)):
            if random.random() < self.prob:
                temp = Image.fromarray(images[i])
                trans_img = temp.transpose(Image.FLIP_LEFT_RIGHT)
                horizontal_images.append(np.array(trans_img))
                horizontal_labels.append(labels[i])
        return np.array(horizontal_images), np.array(horizontal_labels)

    def __vertical_flip(self, images, labels):
        vertical_images, vertical_labels = [], []
        for i in range(len(images)):
            if random.random() < self.prob:
                temp = Image.fromarray(images[i])
                trans_img = temp.transpose(Image.FLIP_TOP_BOTTOM)
                vertical_images.append(np.array(trans_img))
                vertical_labels.append(labels[i])
        return np.array(vertical_images), np.array(vertical_labels)

    def __augment_data(self, images, labels):
        horizontal_flipped_images, horizontal_flipped_labels = self.__horizontal_flip(images, labels)
        vertical_flipped_images, vertical_flipped_labels = self.__vertical_flip(images, labels)

        augmented_images = np.append(horizontal_flipped_images, vertical_flipped_images, axis=0)
        augmented_labels = np.append(horizontal_flipped_labels, vertical_flipped_labels, axis=0)

        augmented_images = np.append(images, augmented_images, axis=0)
        augmented_labels = np.append(labels, augmented_labels, axis=0)

        return augmented_images, augmented_labels
