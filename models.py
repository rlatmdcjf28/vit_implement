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
    def __init__(self, Dimension, batch_size):
        super().__init__()
        self.d = Dimension
        self.cls_tensor = torch.rand(1, 1, self.d)
        self.cls_emb = nn.Parameter(self.cls_tensor)

    def forward(self, x):
        # x is projection : (b, n, d)
        # breakpoint()
        b, n, d = x.shape
        cls_emb = self.cls_emb.expand(b, -1, -1)
        concat = torch.cat([cls_emb, x], dim=1)
        pe = positional_encoding(n+1, d)
        if b > 1:
            pe = pe.unsqueeze(0).expand(b, n+1, d)
        else:
            pe = pe.unsqueeze(0).expand(b, n+1, d)

        return concat
    

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.num_enc_layers)
            ])
        
        self.ln = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):
        for layer in self.encoder:
            out = layer(x)

        output = self.ln(out)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim   = args.hidden_dim,
                                         num_heads   = args.num_heads,
                                         dropout     = args.dropout_rate,
                                         batch_first = True)
        
        self.ln1 = nn.LayerNorm(args.hidden_dim)

        self.ln2 = nn.LayerNorm(args.hidden_dim)
        
        self.ffn1 = nn.Linear(in_features  = args.hidden_dim, 
                              out_features = args.mlp_size)
        
        self.ffn2 = nn.Linear(in_features  = args.mlp_size,
                              out_features = args.hidden_dim)
        
        self.act = nn.GELU()

    def forward(self, x):
        inp = x

        ln = self.ln1(inp)
        mha, _ = self.mha(ln, ln, ln)
        x = mha + inp

        ln = self.ln2(x)
        ffn1 = self.ffn1(ln)
        act = self.act(ffn1)
        ffn2 = self.ffn2(act)

        out = x + ffn2
        
        return out

class FeedForwardNet(nn.Module):
    def __init__(self, in_feature:int, out_feature:int):
        super().__init__()
        self.linear = nn.Linear(in_features  = in_feature,
                                out_features = out_feature)
        self.act = nn.GELU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        inp = x

        x = self.linear(inp)
        x = self.act(x)
        x = self.dropout(x)
        out = x[:, 0]
        
        return out

class ViTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch = Img2Patch(patch_size=args.patch_size)
        self.projection = Projection_layer(Dimension=args.hidden_dim, patch_size=args.patch_size)
        self.class_embedding = ClassEmbedding(Dimension=args.hidden_dim, batch_size=args.batch_size)
        self.transformer_encoder = TransformerEncoder(args)
        self.feedforwardnet = FeedForwardNet(in_feature=args.hidden_dim, 
                                             out_feature=args.num_classes)
       
    def forward(self, x):
        patches = self.patch(x)
        proj = self.projection(patches)
        cls_emb = self.class_embedding(proj)
        te = self.transformer_encoder(cls_emb)
        ffn = self.feedforwardnet(te)
        return ffn


if __name__=='__main__':
    from train import parse
    from torch.utils.data import random_split
    from torchsummary import summary

    args = parse()
 
    args.dataset_dir = '/datasets/cifar10/'

    train_dataset = CustomDataset(root_dir=args.dataset_dir+'train')
    test_dataset  = CustomDataset(root_dir=args.dataset_dir+'test')
    
    train_size = int(len(train_dataset) * 0.8)
    val_size   = int(len(train_dataset) * 0.2)
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_dataloader, val_dataloader, test_dataloader = custom_dataloader(args.batch_size,
                                                                           train_dataset,
                                                                           val_dataset,
                                                                           test_dataset)        
    
    for img, label, cls_idx in train_dataloader:
        break

    model = ViTModel(args)

    patch_size = 8
    hidden_dim = 256
    num_enc_layer = 4
    num_head = 4
    dim_ffn = 512
    num_class = 10

    patch = Img2Patch(patch_size = args.patch_size)
    
    projection = Projection_layer(Dimension  = args.hidden_dim, 
                                  patch_size = args.patch_size)
    
    class_embedding = ClassEmbedding(Dimension  = args.hidden_dim, 
                                     batch_size = args.batch_size)
    
    transformer_encoder = TransformerEncoder(args)
    
    feedforwardnet = FeedForwardNet(in_feature  = args.hidden_dim, 
                                    out_feature = num_class)
    
    breakpoint()

    patches = patch(img)
    proj = projection(patches)
    cls_emb = class_embedding(proj)
    te = transformer_encoder(cls_emb)
    
    breakpoint()

    ffn = feedforwardnet(te)

    breakpoint()

    a = 0