import torch
import torch.nn as nn
from dataset.load_dataset import CustomDataset, custom_dataloader

import numpy as np

class Img2Patch(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.p = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        

    def forward(self, x):
        # x.shape = (b, c, h, w)
        patch = self.unfold(x) # (b, c*p*p, h//p * w//p)
        bs, c, h, w = x.shape

        patches = patch.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        return patches
        

class Projection_layer(nn.Module):
    def __init__(self, Dimension:int, patch_size:int):
        super().__init__()
        self.d = Dimension
        self.p = patch_size
        self.proj_layer = nn.Linear(self.p * self.p * 3, self.d)

    def forward(self, x):
        b, n, c, p, p = x.shape
        x = x.reshape(b, n, c*p*p)
        proj = self.proj_layer(x)
        return proj

def positional_encoding(n, d):
    pe = torch.rand((n, d))
    for i in range(n):
        for j in range(d):
            if j % 2 == 0:
                pe[i][j] = np.sin(i/(10000 ** (j / d)))
            else:
                pe[i][j] = np.cos(i/(10000 ** ((j-1) / d)))

    return pe


class ClassEmbedding(nn.Module):
    def __init__(self, Dimension):
        super().__init__()
        self.d = Dimension
        self.cls_tensor = torch.rand(1, 1, self.d)
        self.cls_emb = nn.Parameter(self.cls_tensor)
        pass

    def forward(self, x):
        # x is projection : (b, n, d)
        b, n, d = x.shape
        concat = torch.cat([self.cls_emb, x], dim=1)
        pe = positional_encoding(n+1, d)

        return concat+pe
    

class TransformerEncoder(nn.Module):
    def __init__(self, num_enc_layer:int, dim:int, num_head:int, dim_ffn:int):
        super().__init__()
        self.te_layer = nn.TransformerEncoderLayer(d_model=dim, 
                                                   nhead=num_head, 
                                                   dim_feedforward=dim_ffn,
                                                   activation='gelu',
                                                   batch_first=True,
                                                   dropout=0.1)
        
        self.te = nn.TransformerEncoder(encoder_layer=self.te_layer,
                                        num_layers=num_enc_layer,
                                        #norm=True
                                        )

    def forward(self, x):
        te = self.te(x)
        return te


class FeedForwardNet(nn.Module):
    def __init__(self, in_feature:int, out_feature:int):
        super().__init__()
        self.linear = nn.Linear(in_features  = in_feature,
                                out_features = out_feature)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x[:, 0]
        return x




if __name__=='__main__':
    train_dir = '/datasets/cifar10/train'
    test_dir  = '/datasets/cifar10/test'

    train_dataset = CustomDataset(root_dir=train_dir)
    test_dataset  = CustomDataset(root_dir=test_dir)

    train_dataloader, test_dataloader = custom_dataloader(batch_size=1, 
                                                        train_dataset=train_dataset, 
                                                        test_dataset=test_dataset)        
    
    for img, label in train_dataloader:
        break

    patch_size = 8
    dim = 512
    num_enc_layer = 4
    num_head = 4
    dim_ffn = 512


    num_class = len(np.unique(train_dataset.labels))
    patch = Img2Patch(patch_size=patch_size)
    projection = Projection_layer(Dimension=dim, patch_size=patch_size)
    class_embedding = ClassEmbedding(Dimension=dim)
    transformer_encoder = TransformerEncoder(num_enc_layer=num_enc_layer,
                                            num_head=num_head,
                                            dim=dim,
                                            dim_ffn=dim_ffn)
    feedforwardnet = FeedForwardNet(in_feature=dim_ffn, out_feature=num_class)
    
    breakpoint()

    patches = patch(img)
    proj = projection(patches)
    cls_emb = class_embedding(proj)
    te = transformer_encoder(cls_emb)
    ffn = feedforwardnet(te)

    breakpoint()

    a = 0