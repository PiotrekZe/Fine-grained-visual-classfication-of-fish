import Dataset as Dataset
import CustomDataset as CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import RunModel
import utils
import CNNModel


def main(num):
    config_data = utils.read_config("config_file.json")

    learning_rate = config_data['model']['learning_rate']
    weight_decay = config_data['model']['weight_decay']
    batch_size = config_data['model']['batch_size']
    epochs = config_data['model']['epochs']
    device = config_data['model']['device']
    attention_map = config_data['model']['attention_map']

    path = config_data['file']['path']
    image_size = config_data['file']['image_size']
    prob_augmentation = config_data['file']['prob_augmentation']
    path_to_save = config_data['file']['path_to_save']

    dataset = Dataset.Dataset(path, image_size, prob_augmentation)
    X_train, X_test, y_train, y_test = dataset.read_dataset()

    train_dataset = CustomDataset.LeafDataset(X_train, y_train)
    test_dataset = CustomDataset.LeafDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CNNModel.FishNet(attention_map).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    run_model = RunModel.RunModel(epochs, device, train_loader, test_loader)

    list_train_loss, list_train_accuracy, list_train_recall, list_train_precision, list_train_f1 = [], [], [], [], []
    list_train_top3, list_train_top5, list_train_top10 = [], [], []
    list_test_loss, list_test_accuracy, list_test_recall, list_test_precision, list_test_f1 = [], [], [], [], []
    list_test_top3, list_test_top5, list_test_top10 = [], [], []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        (train_running_loss, train_accuracy, train_recall, train_precision, train_f1,
         train_top3, train_top5, train_top10) = run_model.train_model(model, criterion, optimizer)

        (test_running_loss, test_accuracy, test_recall, test_precision, test_f1, test_model,
         test_top3, test_top5, test_top10) = run_model.test_model(model, criterion)

        list_train_loss.append(train_running_loss)
        list_train_accuracy.append(train_accuracy)
        list_train_recall.append(train_recall)
        list_train_precision.append(train_precision)
        list_train_f1.append(train_f1)
        list_train_top3.append(train_top3)
        list_train_top5.append(train_top5)
        list_train_top10.append(train_top10)

        list_test_loss.append(test_running_loss)
        list_test_accuracy.append(test_accuracy)
        list_test_recall.append(test_recall)
        list_test_precision.append(test_precision)
        list_test_f1.append(test_f1)
        list_test_top3.append(test_top3)
        list_test_top5.append(test_top5)
        list_test_top10.append(test_top10)

    lists = {
        "train_loss": list_train_loss,
        "train_accuracy": list_train_accuracy,
        "train_recall": list_train_recall,
        "train_precision": list_train_precision,
        "train_f1": list_train_f1,
        "train_top3": list_train_top3,
        "train_top5": list_train_top5,
        "train_top10": list_train_top10,
        "test_loss": list_test_loss,
        "test_accuracy": list_test_accuracy,
        "test_recall": list_test_recall,
        "test_precision": list_test_precision,
        "test_f1": list_test_f1,
        "test_top3": list_test_top3,
        "test_top5": list_test_top5,
        "test_top10": list_test_top10,
    }
    utils.save_model_results(path_to_save, lists, model, num)


if __name__ == '__main__':
    for i in range(40, 50):
        main(i)
