import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from dataset.load_dataset import CustomDataset, custom_dataloader
from models import Img2Patch

train_dir = '/datasets/cifar10/'
test_dir  = '/datasets/cifar10/'

train_dataset = CustomDataset(root_dir=train_dir+'train')
test_dataset  = CustomDataset(root_dir=test_dir+'test')

train_size = int(len(train_dataset) * 0.8)
val_size   = int(len(train_dataset) * 0.2)

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


train_dataloader, val_dataloader, test_dataloader = custom_dataloader(batch_size=1,
                                                                     train_dataset=train_dataset,
                                                                     val_dataset=val_dataset,
                                                                     test_dataset=test_dataset)

def find_key(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

for img, label, cls_idx in train_dataloader:
    cls = find_key(cls_idx, label)
    plt.imshow(img.reshape(3, 32, 32).permute(1, 2, 0).numpy())
    plt.title(cls)
    plt.axis('off')
    # plt.show()
    break

# breakpoint()


patch_size = 8

b, c, h, w = img.shape

patches = Img2Patch(patch_size=patch_size)

patch = patches(img)

breakpoint()

patch_numpy_list = []

for i in range(h//patch_size * w//patch_size):
    patch_numpy_list.append(torch.split(patch, 1, dim=1)[i].reshape(3, 8, 8).permute(1, 2, 0).numpy())

fig, axs = plt.subplots(4, 4, figsize=(4, 4))

for i, ax in enumerate(axs.flat):
    ax.imshow(patch_numpy_list[i])
    ax.axis('off')



plt.suptitle(cls+' img to Patch')
plt.show()


# breakpoint()



a = 0
