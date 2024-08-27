import torch
import torch.nn as nn
from dataset.load_dataset import CustomDataset, custom_dataloader

import numpy as np

class Img2Patch(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.p      = args.patch_size
        self.unfold = nn.Unfold(kernel_size = self.p, 
                                stride      = self.p)
        

    def forward(self, x):
        # x.shape = (b, c, h, w)
        patch       = self.unfold(x) # (b, c*p*p, h//p * w//p)
        bs, c, _, _ = x.shape

        patches = patch.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)

        return patches # patches.shape = (b, n, p, p, c)
        

class Projection_layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d          = args.hidden_dim
        self.p          = args.patch_size
        self.proj_layer = nn.Linear(self.p * self.p * 3, self.d)

    def forward(self, x):
        b, n, c, p, p = x.shape
        x             = x.reshape(b, n, c*p*p)
        proj          = self.proj_layer(x)

        return proj # proj.shape = (b, n, h_d)



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
    def __init__(self, args):
        super().__init__()
        self.d          = args.hidden_dim
        self.device     = args.device
        self.cls_tensor = torch.rand(1, 1, self.d)
        self.cls_emb    = nn.Parameter(self.cls_tensor)

    def forward(self, x):
        # x.shape = (b, n, d)
        # breakpoint()
        b, n, d = x.shape
        cls_emb = self.cls_emb.expand(b, -1, -1)
        concat  = torch.cat([cls_emb, x], dim=1)
        pe      = positional_encoding(n+1, d).unsqueeze(0).expand(b, n+1, d).to(self.device)
        
        return concat + pe # shape = (b, n+1, d)
    

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(args.num_enc_layers)
            ])
        
        self.ln = nn.LayerNorm(args.hidden_dim)

    def forward(self, x):
        out = x

        for layer in self.encoder:
            out = layer(out)

        output = self.ln(out)
        return output # output.shape = (b, n+1, d)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha  = nn.MultiheadAttention(embed_dim   = args.hidden_dim,
                                          num_heads   = args.num_heads,
                                          dropout     = args.dropout_rate,
                                          batch_first = True)
        
        self.ln1  = nn.LayerNorm(args.hidden_dim)

        self.ln2  = nn.LayerNorm(args.hidden_dim)
        
        self.ffn1 = nn.Linear(in_features  = args.hidden_dim, 
                              out_features = args.mlp_size)
        
        self.ffn2 = nn.Linear(in_features  = args.mlp_size,
                              out_features = args.hidden_dim)
        
        self.act  = nn.GELU()

    def forward(self, x):
        inp    = x # shape = (b, n+1, d)

        ln     = self.ln1(inp)
        mha, _ = self.mha(ln, ln, ln)
        x      = mha + inp

        ln     = self.ln2(x)
        
        ffn1   = self.ffn1(ln)  # shape = (b, n+1, mlp_size)
        act    = self.act(ffn1) 
        ffn2   = self.ffn2(act) # shape = (b, n+1, d)

        out    = x + ffn2
        
        return out # shape = (b, n+1, d)

class FeedForwardNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear  = nn.Linear(in_features  = args.hidden_dim,
                                 out_features = args.num_classes)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(args.dropout_rate)
        self.ln      = nn.LayerNorm(args.num_classes)

    def forward(self, x):
        inp = x # shape = (b, n+1, d)

        x   = self.linear(inp)
        x   = self.act(x)
        x   = self.dropout(x)
        x   = x[:, 0, :] # shape = (b, num_classes)
        out = self.ln(x)

        return out # shape = (b, num_classes)

class ViTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch               = Img2Patch(args)

        self.projection          = Projection_layer(args)
        
        self.class_embedding     = ClassEmbedding(args)
        
        self.transformer_encoder = TransformerEncoder(args)

        self.feedforwardnet      = FeedForwardNet(args)
       
    def forward(self, x):
        # x.shape = (b, 3, 32, 32)
        patches = self.patch(x)
        # patches.shape = (b, n, c, p, p)
        proj    = self.projection(patches)
        # proj.shape = (b, n, hidden_dim)
        cls_emb = self.class_embedding(proj)
        # cls_emb.shape = (b, n+1, hidden_dim)
        te      = self.transformer_encoder(cls_emb)
        # te.shape = (n, n+1, hidden_dim)
        ffn     = self.feedforwardnet(te)
        return ffn # ffn.shape = (b, num_classes)


if __name__=='__main__':
    from train import parse
    from torch.utils.data import random_split
    from torchsummary import summary

    args = parse()
 
    args.dataset_dir = '/Data_RTX4090_server/datasets/cifar10/'

    train_dataset = CustomDataset(root_dir=args.dataset_dir+'train')
    test_dataset  = CustomDataset(root_dir=args.dataset_dir+'test')
    
    train_size = int(len(train_dataset) * 0.8)
    val_size   = int(len(train_dataset) * 0.2)
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


    train_dataloader, val_dataloader, test_dataloader = custom_dataloader(args.batch_size,
                                                                          train_dataset,
                                                                          val_dataset,
                                                                          test_dataset)        
    
    for img, label in train_dataloader:
        break

    model = ViTModel(args)

    patch_size = 8
    hidden_dim = 256
    num_enc_layer = 4
    num_head = 4
    dim_ffn = 512
    num_class = 10

    patch = Img2Patch(args)
    
    projection = Projection_layer(args)
    
    class_embedding = ClassEmbedding(args)
    
    transformer_encoder = TransformerEncoder(args)
    
    feedforwardnet = FeedForwardNet(args)
    
    breakpoint()

    patches = patch(img)
    proj = projection(patches)
    cls_emb = class_embedding(proj)
    te = transformer_encoder(cls_emb)
    
    breakpoint()

    ffn = feedforwardnet(te)

    breakpoint()

    a = 0