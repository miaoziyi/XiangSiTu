import cv2
import torch
from torch.utils.data import Dataset

class ImgDataset(Dataset):
    def __init__(self, csv, transforms=None):
        self.csv = csv.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        #         text = row.title

        image = cv2.imread(row.filepath)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(row.filepath)
            # exit(1)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(row.spuId)