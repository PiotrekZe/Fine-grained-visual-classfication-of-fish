import CNNModel
import Dataset
import CustomDataset
import RunModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import utils


def main():
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

    train_dataset = CustomDataset.CroatianFish(X_train, y_train)
    test_dataset = CustomDataset.CroatianFish(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CNNModel.FishNet(attention_map).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    run_model = RunModel.RunModel(epochs, device, train_loader, test_loader)


    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        train_running_loss, train_accuracy, train_recall, train_precision, train_f1 = run_model.train_model(model,
                                                                                                            criterion,
                                                                                                            optimizer)
        test_running_loss, test_accuracy, test_recall, test_precision, test_f1 = run_model.test_model(model, criterion)


if __name__ == '__main__':
    main()
