import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
import numpy as np
from pdb import set_trace as pause
from field_components.encodings import NeRFEncoding
import tinycudann as tcnn


class SDF2Network(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=2,
                 skips=[],
                 hidden_dim=256,
                 clip_sdf=None,
                 n_encoders=1,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        self.max_res = 2048
        self.base_res = 16
        self.num_levels = 16
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        # self.encoder, self.in_dim = get_encoder(encoding)
        self.encoders = []
        self.n_encoders = n_encoders
        for i in range(n_encoders):
            if False:
                encoder, self.in_dim = get_encoder(encoding,
                    input_dim=3, 
                    multires=6, 
                    degree=4,
                    # num_levels=16, level_dim=2, base_resolution=64, log2_hashmap_size=22,
                    # desired_resolution=4096, 
                    num_levels=16, level_dim=4, base_resolution=16, log2_hashmap_size=18,
                    desired_resolution=2048, 
                    # num_levels=16, level_dim=2, base_resolution=64, log2_hashmap_size=19,
                    # desired_resolution=4096, 
                    align_corners=False,
                    )
            else:
                encoder = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": self.num_levels,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 22,
                        "base_resolution": self.base_res,
                        "per_level_scale": self.growth_factor,
                        "interpolation": "Linear",  #"Smoothstep" if smoothstep else "Linear",
                    },
                )
                self.in_dim = self.num_levels * 2
            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)
        # self.encoder = self.encoders[0]
        position_encoding_max_degree = 6
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=position_encoding_max_degree - 1,
            include_input=False,
            off_axis=False,
        )
        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + 3 + self.position_encoding.get_out_dim()  
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            lin = nn.Linear(in_dim, out_dim, bias=True)
            # geometric initialization
            if l == num_layers - 1:
                torch.nn.init.constant_(lin.bias, 0.8)
                torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
            elif l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            lin = nn.utils.weight_norm(lin)
            backbone.append(lin)

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, x, encoder_id=None, return_features=False):
        # x: [B, 3]
        if encoder_id is None:
            x = [torch.cat((x[:,i]*2, self.position_encoding(x[:,i]*2), self.encoders[i]((x[:, i]+1)/2)), -1) for i in range(self.n_encoders)] 
        else:
            x = [torch.cat((x[:,0]*2, self.position_encoding(x[:,0]*2), self.encoders[i]((x[:, 0]+1)/2)), -1) for i in [encoder_id]] 
        x = torch.cat(x, 0)
        if return_features:
            return x

        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                # h = F.relu(h, inplace=True)
                h = F.softplus(h, beta=100)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h

class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 n_encoders=1,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        # self.encoder, self.in_dim = get_encoder(encoding)
        self.encoders = []
        self.n_encoders = n_encoders
        self.max_res = 2048
        self.base_res = 16
        self.num_levels = 16
        self.features_per_level = 4
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))
        for i in range(n_encoders):
            if True:
                encoder, self.in_dim = get_encoder(encoding,
                    input_dim=3, 
                    multires=6, 
                    degree=4,
                    # num_levels=16, level_dim=2, base_resolution=64, log2_hashmap_size=22,
                    # desired_resolution=4096, 
                    num_levels=self.num_levels, level_dim=self.features_per_level, 
                    base_resolution=self.base_res, log2_hashmap_size=22,
                    desired_resolution=self.max_res, 
                    # num_levels=16, level_dim=2, base_resolution=64, log2_hashmap_size=19,
                    # desired_resolution=4096, 
                    align_corners=False,
                    )
            else:
                encoder = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": self.num_levels,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 22,
                        "base_resolution": self.base_res,
                        "per_level_scale": self.growth_factor,
                        "interpolation": "Linear",  #"Smoothstep" if smoothstep else "Linear",
                    },
                )
                self.in_dim = self.num_levels * 2
            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)
        self.encoder = self.encoders[0]
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

    
    def forward(self, x, encoder_id=None, return_features=False):
        # x: [B, 3]
        # x = torch.ones_like(x)[:90000]
        # x[:,0, 0] *= torch.arange(90000).cuda() / 90000 - 0.5
        # x[:,0, 1] *= -torch.arange(90000).cuda() / 90000
        # x[:,0, 2] *= 1-torch.arange(90000).cuda() / 90000

        if encoder_id is None:
            x = [self.encoders[i](x[:, i]) for i in range(self.n_encoders)] 
        else:
            x = [self.encoders[i](x[:, 0]) for i in [encoder_id]] 
        x = torch.cat(x, 0).float()
        if return_features:
            return x

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
