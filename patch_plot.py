import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from dataset.load_dataset import CustomDataset, custom_dataloader
from models import Img2Patch

'''
inp_img = torch.zeros((1, 3, 16, 16))
cc, hh, ww = inp_img.shape[1], inp_img.shape[2], inp_img.shape[3]

for channel in range(3):
    start_value = 1 + channel * 16 
    for row in range(16):
        inp_img[0, channel, row, :] = start_value + row

train_dir = '/datasets/cifar10/train'

train_dataset = CustomDataset(train_dir)

train_img = train_dataset.__getitem__(1)
totensor = ToTensor()

train_img = totensor(train_img)
train_img_tensor = train_img.reshape(1, 3, 32, 32)
cc, hh, ww = train_img_tensor.shape[1], train_img_tensor.shape[2], train_img_tensor.shape[3]
train_img_np = train_img_tensor.reshape(3, 32, 32).permute(1, 2, 0).numpy()

patchfy = Img2Patch(patch_size=pp)

patches = patchfy(train_img_tensor) # (1, pp, hh//pp, ww//pp)

patches_numpy_list = []

for i in range(hh//pp * ww//pp):
    patches_numpy_list.append(torch.split(patches, 1, dim=1)[i].reshape(3, 8, 8).permute(1, 2, 0).numpy())

fig, axs = plt.subplots(4, 4, figsize=(4, 4))

for i, ax in enumerate(axs.flat):
    ax.imshow(patches_numpy_list[i])
    ax.axis('off')

plt.show()
'''

train_dir = '/datasets/cifar10/train'
test_dir  = '/datasets/cifar10/test'

train_dataset = CustomDataset(root_dir=train_dir)
test_dataset  = CustomDataset(root_dir=test_dir)

train_dataloader, test_dataloader = custom_dataloader(batch_size=1, 
                                                      train_dataset=train_dataset, 
                                                      test_dataset=test_dataset)

for img, label in train_dataloader:
    plt.imshow(img.reshape(3, 32, 32).permute(1, 2, 0).numpy())
    plt.title(label[0])
    plt.axis('off')
    # plt.show()
    break

patch_size = 8

b, c, h, w = img.shape

patches = Img2Patch(patch_size=patch_size)

patch = patches(img)

patch_numpy_list = []

for i in range(h//patch_size * w//patch_size):
    patch_numpy_list.append(torch.split(patch, 1, dim=1)[i].reshape(3, 8, 8).permute(1, 2, 0).numpy())

fig, axs = plt.subplots(4, 4, figsize=(4, 4))

for i, ax in enumerate(axs.flat):
    ax.imshow(patch_numpy_list[i])
    ax.axis('off')

plt.suptitle(label[0]+' img to Patch')
plt.show()


# breakpoint()



a = 0
