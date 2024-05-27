import numpy as np
import torch
import utils


class RunModel:
    def __init__(self, epochs, device, train_loader, test_loader):
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, model, criterion, optimizer):
        model.train()
        running_loss = 0
        predictions_list = []
        targets_list = []
        main_idx = 0
        top3_idx, top5_idx, top10_idx = 0, 0, 0
        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            batch_outputs = model(images)
            loss = criterion(batch_outputs, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            running_loss += loss.item()

            top3_values, top3_indices = torch.topk(batch_outputs, 3)
            top5_values, top5_indices = torch.topk(batch_outputs, 5)
            top10_values, top10_indices = torch.topk(batch_outputs, 10)

            _, batch_outputs = torch.max(batch_outputs, 1)
            predictions_list.append(np.array(batch_outputs.cpu().detach()))
            targets_list.append(np.array(targets.cpu().detach()))

            for i in range(len(top3_indices)):
                main_idx += 1
                if targets[i] in top3_indices[i]:
                    top3_idx += 1
                if targets[i] in top5_indices[i]:
                    top5_idx += 1
                if targets[i] in top10_indices[i]:
                    top10_idx += 1

        targets_list = np.concatenate(targets_list)
        predictions_list = np.concatenate(predictions_list)
        running_loss = running_loss / len(self.train_loader)
        accuracy, recall, precision, f1 = utils.calculate_metrics(targets_list, predictions_list)

        print(f"Training. Loss: {running_loss}. Accuracy: {accuracy}, "
              f"Top3: {top3_idx / main_idx}, Top5: {top5_idx / main_idx}, Top10: {top10_idx / main_idx}")
        return (running_loss, accuracy, recall, precision, f1,
                top3_idx / main_idx, top5_idx / main_idx, top10_idx / main_idx)

    def test_model(self, model, criterion):
        model.eval()
        running_loss = 0
        predictions_list = []
        targets_list = []
        main_idx = 0
        top3_idx, top5_idx, top10_idx = 0, 0, 0
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                batch_outputs = model(images)
                loss = criterion(batch_outputs, targets)

                running_loss += loss.item()

                top3_values, top3_indices = torch.topk(batch_outputs, 3)
                top5_values, top5_indices = torch.topk(batch_outputs, 5)
                top10_values, top10_indices = torch.topk(batch_outputs, 10)

                _, batch_outputs = torch.max(batch_outputs, 1)
                predictions_list.append(np.array(batch_outputs.cpu().detach()))
                targets_list.append(np.array(targets.cpu().detach()))

                for i in range(len(top3_indices)):
                    main_idx += 1
                    if targets[i] in top3_indices[i]:
                        top3_idx += 1
                    if targets[i] in top5_indices[i]:
                        top5_idx += 1
                    if targets[i] in top10_indices[i]:
                        top10_idx += 1

        targets_list = np.concatenate(targets_list)
        predictions_list = np.concatenate(predictions_list)
        running_loss = running_loss / len(self.train_loader)
        accuracy, recall, precision, f1 = utils.calculate_metrics(targets_list, predictions_list)

        print(f"Testing. Loss: {running_loss}. Accuracy: {accuracy}, "
              f"Top3: {top3_idx / main_idx}, Top5: {top5_idx / main_idx}, Top10: {top10_idx / main_idx}")
        return (running_loss, accuracy, recall, precision, f1, model,
                top3_idx / main_idx, top5_idx / main_idx, top10_idx / main_idx)
