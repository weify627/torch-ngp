import numpy as np

import torch
from torch.utils.data import Dataset

import trimesh
import pysdf
from pdb import set_trace as pause

def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    value = (value - vmin) / (vmax - vmin) # range in [0, 1]
    cmap = cm.get_cmap(cmap_name) 
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb

def plot_pointcloud(pc, sdfs):
    # pc: [N, 3]
    # sdfs: [N, 1]
    color = map_color(sdfs.squeeze(1))
    pc = trimesh.PointCloud(pc, color)
    trimesh.Scene([pc]).show()    

import struct
def read_sdf(filename):
    # Define the format string for unpacking the floats
    # '<' for little-endian, '4f' for four floats
    format_str = '<4f' 

    # Create an empty list to store the read values
    data = []

    # Open the file in binary mode
    with open(filename, 'rb') as file:
        while True:
            # Read 4 floats (16 bytes) at a time
            bytes = file.read(16)

            # Break the loop if we've reached the end of the file
            if not bytes:
                break

            # Unpack the bytes and append to the data list
            data.append(struct.unpack(format_str, bytes))

    return np.array(data)


# SDF dataset
class SDF3Dataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        # self.mesh = trimesh.load(path, force='mesh')
        # sdf_fname = self.config.data / "rand_surf-4m.ply"
        self.mesh = trimesh.load(path, preprocess=False)
        sdf_onsurface = self.mesh.vertices
        if True:
            sdf_fname = f"{path.split('/rand')[0]}/near_surf-400k.sdf"
            sdf_offsurface = read_sdf(sdf_fname)
            sdf_offsurface = sdf_offsurface[np.abs(sdf_offsurface[:, 3])<1]
            n_onsurface = sdf_onsurface.shape[0]
            n_offsurface = sdf_offsurface.shape[0]
            sdf_samples = np.zeros((n_onsurface + n_offsurface, 4))
            sdf_samples[:n_onsurface, :3] = sdf_onsurface
            sdf_samples[n_onsurface:] = sdf_offsurface
        else:
            n_onsurface = sdf_onsurface.shape[0]
            n_offsurface = 0
            sdf_samples = np.zeros((n_onsurface + n_offsurface, 4))
            sdf_samples[:n_onsurface, :3] = sdf_onsurface

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        # vs = self.mesh.vertices
        vs = sdf_samples[..., :3]
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        self.sdf_samples = sdf_samples
        pause()
        self.sdf_samples[..., :3] = vs
        self.sdf_samples[..., 3] *= v_scale
        self.n_sdf_samples = sdf_samples.shape[0]
        choices = np.arange(self.n_sdf_samples)
        np.random.shuffle(choices)
        self.sdf_samples = self.sdf_samples[choices]

        # print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        # if not self.mesh.is_watertight:
            # print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        # self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        # online sampling
        # sdfs = np.zeros((self.num_samples, 1))
        # # surface
        # points_surface = self.mesh.sample(self.num_samples * 7 // 8)
        # # perturb surface
        # points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        # # random
        # points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        # points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        # sdfs[self.num_samples // 2:] = -self.sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)
        start = np.random.choice(self.n_sdf_samples-1)
        sdf_samples = np.concatenate([self.sdf_samples[start:], self.sdf_samples[:start]], 0)
        interval = 1 + np.random.choice(self.n_sdf_samples // self.num_samples - 2)
        sampled = sdf_samples[::interval][:self.num_samples].astype(np.float32)
        sdfs = -sampled[..., 3:]
        points = sampled[..., :3]

 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        #plot_pointcloud(points, sdfs)

        return results
# SDF dataset
class SDF2Dataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        # self.mesh = trimesh.load(path, force='mesh')
        # sdf_fname = self.config.data / "rand_surf-4m.ply"
        self.mesh = trimesh.load(path, preprocess=False)
        sdf_onsurface = self.mesh.vertices
        if True:
            sdf_fname = f"{path.split('/rand')[0]}/near_surf-400k.sdf"
            sdf_offsurface = read_sdf(sdf_fname)
            sdf_offsurface = sdf_offsurface[np.abs(sdf_offsurface[:, 3])<1]
            n_onsurface = sdf_onsurface.shape[0]
            n_offsurface = sdf_offsurface.shape[0]
            sdf_samples = np.zeros((n_onsurface + n_offsurface, 4))
            sdf_samples[:n_onsurface, :3] = sdf_onsurface
            sdf_samples[n_onsurface:] = sdf_offsurface
        else:
            n_onsurface = sdf_onsurface.shape[0]
            n_offsurface = 0
            sdf_samples = np.zeros((n_onsurface + n_offsurface, 4))
            sdf_samples[:n_onsurface, :3] = sdf_onsurface

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        # vs = self.mesh.vertices
        vs = sdf_samples[..., :3]
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        self.sdf_samples = sdf_samples
        pause()
        self.sdf_samples[..., :3] = vs
        self.sdf_samples[..., 3] *= v_scale
        self.n_sdf_samples = sdf_samples.shape[0]
        choices = np.arange(self.n_sdf_samples)
        np.random.shuffle(choices)
        self.sdf_samples = self.sdf_samples[choices]

        # print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        # if not self.mesh.is_watertight:
            # print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        # self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        # online sampling
        # sdfs = np.zeros((self.num_samples, 1))
        # # surface
        # points_surface = self.mesh.sample(self.num_samples * 7 // 8)
        # # perturb surface
        # points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
        # # random
        # points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
        # points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        # sdfs[self.num_samples // 2:] = -self.sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)
        start = np.random.choice(self.n_sdf_samples-1)
        sdf_samples = np.concatenate([self.sdf_samples[start:], self.sdf_samples[:start]], 0)
        interval = 1 + np.random.choice(self.n_sdf_samples // self.num_samples - 2)
        sampled = sdf_samples[::interval][:self.num_samples].astype(np.float32)
        sdfs = -sampled[..., 3:]
        points = sampled[..., :3]

 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        #plot_pointcloud(points, sdfs)

        return results

class SDFDataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None):
        super().__init__()
        self.path = path

        # load obj 
        self.mesh = trimesh.load(path, force='mesh')

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        vs = self.mesh.vertices
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        vs = (vs - v_center[None, :]) * v_scale
        self.mesh.vertices = vs

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size

    
    def __len__(self):
        return self.size

    def __getitem__(self, _):

        # online sampling
        sdfs = np.zeros((self.num_samples, 1))
        # surface
        if False:
            points_surface = self.mesh.sample(self.num_samples * 7 // 8)
            # perturb surface
            points_surface[self.num_samples // 2:] += 0.01 * np.random.randn(self.num_samples * 3 // 8, 3)
            # random
            points_uniform = np.random.rand(self.num_samples // 8, 3) * 2 - 1
            points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

            sdfs[self.num_samples // 2:] = -self.sdf_fn(points[self.num_samples // 2:])[:,None].astype(np.float32)
        else:
            points_surface = self.mesh.sample(self.num_samples // 2)
            # perturb surface
            points_surface[self.num_samples // 4:] += 0.01 * np.random.randn(self.num_samples // 4, 3)
            # random
            points_uniform = np.random.rand(self.num_samples // 2, 3) * 2 - 1
            points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

            sdfs[self.num_samples // 4:] = -self.sdf_fn(points[self.num_samples // 4:])[:,None].astype(np.float32)
 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        #plot_pointcloud(points, sdfs)

        return results
