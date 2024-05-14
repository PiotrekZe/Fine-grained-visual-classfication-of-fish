import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2


class LeafDataset(Dataset):
    def __init__(self, data, labels, size):
        self.data = data
        self.labels = labels
        self.size = size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert("RGB")
        label = self.labels[index]

        if self.size is not None:
            image = image.resize((self.size, self.size), resample=Image.BILINEAR)

        image = cv2.addWeighted(src1=np.array(image), alpha=1.5,
                                src2=np.zeros(np.array(image).shape, np.array(image).dtype), beta=0, gamma=0)

        image = self.transform(np.array(image))

        return image, torch.tensor(label, dtype=torch.long)
