import torch
import torch.nn as nn
from dataset.load_dataset import CustomDataset


class Patchfy(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        
        pass
    def forward(self, x):
        # x.shape = (b, c, h, w)
        bs, c, h, w = x.shape

        patch = self.unfold(x)
        
        patches = patch.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)

        return patches
        
        
        pass




if __name__=='__main__':
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt

    pp = 8
    '''
    inp_img = torch.zeros((1, 3, 16, 16))
    cc, hh, ww = inp_img.shape[1], inp_img.shape[2], inp_img.shape[3]

    for channel in range(3):
        start_value = 1 + channel * 16 
        for row in range(16):
            inp_img[0, channel, row, :] = start_value + row
    '''
    train_dir = '/datasets/cifar10/train'
    
    train_dataset = CustomDataset(train_dir)

    train_img = train_dataset.__getitem__(1)
    totensor = ToTensor()

    train_img = totensor(train_img)
    train_img_tensor = train_img.reshape(1, 3, 32, 32)
    cc, hh, ww = train_img_tensor.shape[1], train_img_tensor.shape[2], train_img_tensor.shape[3]
    train_img_np = train_img_tensor.reshape(3, 32, 32).permute(1, 2, 0).numpy()

    patchfy = Patchfy(patch_size=pp)

    patches = patchfy(train_img_tensor) # (1, pp, hh//pp, ww//pp)
    
    patches_numpy_list = []

    for i in range(hh//pp * ww//pp):
        patches_numpy_list.append(torch.split(patches, 1, dim=1)[i].reshape(3, 8, 8).permute(1, 2, 0).numpy())
    
    fig, axs = plt.subplots(4, 4, figsize=(4, 4))

    for i, ax in enumerate(axs.flat):
        ax.imshow(patches_numpy_list[i])
        ax.axis('off')
        
    breakpoint()

    a = 0