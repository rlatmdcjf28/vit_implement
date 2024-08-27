import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=True, Normalize=False):
        self.root_dir = root_dir
        if transform == True:
            if Normalize == False:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                    ])
            else:
                self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform=None
        
        self.images = []
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def class_names_list(self):
        keys = []
        for key in self.class_to_idx:
            keys.append(key)

        return keys
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label

def custom_dataloader(batch_size, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader



if __name__=='__main__':
    train_img_dir = '/datasets/cifar10/train'
    val_img_dir = '/datasets/cifar10/test'

    train_dataset = CustomDataset(train_img_dir)
    test_dataset = CustomDataset(val_img_dir)

    train_dataloader, test_dataloader = custom_dataloader(train_dataset=train_dataset, 
                                                          test_dataset=test_dataset, 
                                                          batch_size=1)
    for img, label in train_dataloader:
        print(img.shape)
        print(label[0])
        break

    breakpoint()

    a = 0

    pass