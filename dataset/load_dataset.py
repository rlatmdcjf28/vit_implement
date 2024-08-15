import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms

'''
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir : Image Directory
            transform : 샘플에 적용될 Optional transform
        """
        self.image_dir = image_dir
        self.transform = transform
        self.labels = os.listdir(image_dir)
        self.image_filenames = [x for x in os.listdir(image_dir+'/'+self.labels[0]) if x.endswith('.jpg') or x.endswith('.png')]


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir+'/'+self.labels[0], self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')  # 이미지를 RGB로 변환
        
        if self.transform:
            image = self.transform(image)

        return image
'''
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=True, Normalize=False):
        """
        Args:
            root_dir (string): 데이터셋의 최상위 디렉토리
            transform (callable, optional): 샘플에 적용될 변환
        """
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
        self.labels = []
        
        # 이미지 파일과 레이블 읽기
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(root.split('/')[-1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, label

def custom_dataloader(batch_size, train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader



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
