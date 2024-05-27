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
        folders = os.listdir(self.path)

        for idx, folder in enumerate(folders):
            folder_path = os.path.join(self.path, folder)
            files = os.listdir(folder_path)
            for file in files:
                # file_path = os.path.join(folder_path, file)
                # images.append(file_path)
                file_path = os.path.join(folder_path, file)
                image = Image.open(file_path).convert("RGB")
                image = image.resize((self.size, self.size), resample=Image.BILINEAR)
                # image = cv2.addWeighted(src1=np.array(image), alpha=1.5,
                #                         src2=np.zeros(np.array(image).shape, np.array(image).dtype), beta=0, gamma=0)
                images.append(np.array(image))
                targets.append(idx)

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
