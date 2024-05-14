import json
import os
import torch

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np


def calculate_metrics(targets_list, predictions_list):
    accuracy = accuracy_score(targets_list, predictions_list)
    recall = recall_score(targets_list, predictions_list, average='macro', zero_division=0)
    precision = precision_score(targets_list, predictions_list, average='macro', zero_division=0)
    f1 = f1_score(targets_list, predictions_list, average='macro', zero_division=0)

    return accuracy, recall, precision, f1


def create_plot(train_data, test_data, plot_name):
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    plt.figure(figsize=(8, 6))
    plt.plot((range(len(train_data))), train_data, label="Train set", linewidth=2)
    plt.plot((range(len(test_data))), test_data, label="Test set", linewidth=2)
    plt.legend(loc="lower right")
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel(f'{plot_name}')
    if plot_name != "Loss value":
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    return plt


def save_model_results(path, lists, model, num):
    path_to_save = os.path.join(path, str(num))
    os.mkdir(path_to_save)
    list_train_loss = lists["train_loss"]
    list_train_accuracy = lists["train_accuracy"]
    list_train_recall = lists["train_recall"]
    list_train_precision = lists["train_precision"]
    list_train_f1 = lists["train_f1"]
    list_train_top3 = lists["train_top3"]
    list_train_top5 = lists["train_top5"]
    list_train_top10 = lists["train_top10"]
    list_test_loss = lists["test_loss"]
    list_test_accuracy = lists["test_accuracy"]
    list_test_recall = lists["test_recall"]
    list_test_precision = lists["test_precision"]
    list_test_f1 = lists["test_f1"]
    list_test_top3 = lists["test_top3"]
    list_test_top5 = lists["test_top5"]
    list_test_top10 = lists["test_top10"]
    for key, val in lists.items():
        with open(f"{path_to_save}/{key}.txt", "w") as f:
            for el in val:
                f.write(f"{el};")

    torch.save(model.state_dict(), os.path.join(path_to_save, "model.pt"))


def read_config(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data
