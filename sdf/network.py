import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        # self.encoder, self.in_dim = get_encoder(encoding)
        self.encoder, self.in_dim = get_encoder(encoding,
                input_dim=3, 
                multires=6, 
                degree=4,
                # num_levels=16, level_dim=2, base_resolution=64, log2_hashmap_size=22,
                # desired_resolution=4096, 
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=22,
                desired_resolution=2048, 
                # num_levels=16, level_dim=2, base_resolution=64, log2_hashmap_size=19,
                # desired_resolution=4096, 
                align_corners=False,
                )

        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, x):
        # x: [B, 3]

        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
