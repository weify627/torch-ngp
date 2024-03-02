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
def read_sdf_slow(filename, normal=False):
    # Define the format string for unpacking the floats
    # '<' for little-endian, '4f' for four floats
    format_str = '<7f' if normal else '<4f' 

    # Create an empty list to store the read values
    data = []

    # Open the file in binary mode
    with open(filename, 'rb') as file:
        while True:
            # Read 4 floats (16 bytes) at a time
            if normal:
                bytes = file.read(28)
            else:
                bytes = file.read(16)

            # Break the loop if we've reached the end of the file
            if not bytes:
                break

            # Unpack the bytes and append to the data list
            data.append(struct.unpack(format_str, bytes))

    return np.array(data)

def read_sdf(filename, normal=False):
    """
    Read floats from a binary file into an Nx4 numpy array efficiently.

    Args:
    - filename: String, the path to the binary file to read.

    Returns:
    - Nx4 numpy array of the read floats.
    """
    # Open the file in binary read mode
    with open(filename, 'rb') as file:
        # Read the entire file content into a bytes object
        data = file.read()

    # Convert the bytes data into a 1D numpy array of float32
    # Assuming the file contains only float32 values
    array = np.frombuffer(data, dtype=np.float32)

    # Reshape the array to Nx4 since we know each point consists of 4 floats
    # The '-1' in reshape automatically calculates the size of the first dimension
    array = array.reshape(-1, 7 if normal else 4)

    return array


def write_sdf(filename, array):
    """
    Save an Nx4 numpy array to a .bin file in a fast manner.

    Args:
    - filename: String, the name of the file to save.
    - array: Nx4 numpy array.
    """
    # Ensure the array is of the right shape and type
    if array.shape[1] != 4:
        raise ValueError("Array must be Nx4 and of type float32")

    # Open the file in binary write mode
    with open(filename, 'wb') as file:
        # Write the entire array to file in one go
        file.write(array.astype(np.float32).tobytes())

# SDF dataset for 400M points
class SDF5Dataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None, part=0, scene_list=[], dummy=False):
        super().__init__()
        self.path = path
        if dummy:
            self.mesh = trimesh.load(path, process=False)
            vs = self.mesh.vertices
            vmin = vs.min(0)
            vmax = vs.max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            # vs = (vs - v_center[None, :]) * v_scale
            self.v_center = v_center
            self.v_scale = v_scale
            self.transform = np.eye(4)
            self.transform *= v_scale
            self.transform[:3, 3] = -v_center * v_scale
            self.transform[3, 3] = 1
            vs = (vs - v_center[None, :]) * v_scale
            self.bounds_min = 0
            self.bounds_max = 0
            self.pts = vs
            return
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
        self.scene_list = scene_list
        self.num_scenes = len(scene_list)
        self.scenes = [SDF4Dataset(path, size, num_samples, clip_sdf, part, scene, dummy) for scene in scene_list]
        self.transform = [scene.transform for scene in self.scenes]
        self.bounds_min = [scene.bounds_min for scene in self.scenes]
        self.bounds_max = [scene.bounds_max for scene in self.scenes]

    def load_data(self, part=0):
        for i in range(self.num_scenes):
            self.scenes[i].load_data(part)

    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        results = []
        for i in range(self.num_scenes):
            i_results = self.scenes[i].__getitem__(idx)
            results.append(i_results)

        results = {
            'sdfs': np.concatenate([result["sdfs"] for result in results], 0),
            'points': np.stack([result["points"] for result in results], 1),
        }


        return results

# SDF dataset for 400M points
class SDF4Dataset(Dataset):
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None, part=0, scene="", dummy=False):
        super().__init__()
        self.path = f"{path}/{scene}/scans/rand_surf-40m-v9"
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
        self.load_data(part)
        self.bounds_min = torch.FloatTensor(self.sdf_samples[..., :3].min(0))
        self.bounds_max = torch.FloatTensor(self.sdf_samples[..., :3].max(0))
        if dummy:
            self.sdf_samples = self.sdf_samples[:2]

    def load_data(self, part=0):

        # load obj 
        path = self.path
        # if path is None:
            # path = self.path
        # else:
            # self.path = path
        path = f"{path}-{part}.ply"
        mesh = trimesh.load(path, process=False)
        sdf_onsurface = mesh.vertices.astype(np.float32)
        #TODO: check bbox vs off_surface
        if True:
            sdf_fname = path.replace("rand_surf", "near_surf")[:-3].replace("-cur1", "").replace("v9-", "exp5-") + "sdf"
            sdf_offsurface = read_sdf(sdf_fname)
            n_onsurface = sdf_onsurface.shape[0]
            n_offsurface = sdf_offsurface.shape[0]
            sdf_samples = np.zeros((n_onsurface + n_offsurface, 4)).astype(np.float32)
            sdf_samples[:n_onsurface, :3] = sdf_onsurface
            sdf_samples[n_onsurface:] = sdf_offsurface

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        # vs = self.mesh.vertices
        vs_full = sdf_samples[..., :3]
        vs = sdf_samples[:n_onsurface, :3]
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        # vs = (vs - v_center[None, :]) * v_scale
        if part == 0:
            self.v_center = v_center
            self.v_scale = v_scale
            self.transform = np.eye(4)
            self.transform *= v_scale
            self.transform[:3, 3] = -v_center * v_scale
            self.transform[3, 3] = 1
        else:
            v_center = self.v_center
            v_scale = self.v_scale
        vs_full = (vs_full - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        self.sdf_samples = sdf_samples
        # self.sdf_samples[..., :3] = vs
        self.sdf_samples[..., :3] = vs_full
        self.sdf_samples[..., 3] *= v_scale
        self.n_sdf_samples = sdf_samples.shape[0]

        choices = np.arange(self.n_sdf_samples)
        if self.size == 1:
            choices = choices[::100][:self.num_samples]
        else:
            np.random.shuffle(choices)
        self.sdf_samples = self.sdf_samples[choices]

        print(f"loaded {path}, {sdf_fname}")

    
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
        if False:
            start = np.random.choice(self.n_sdf_samples-1)
            sdf_samples = np.concatenate([self.sdf_samples[start:], self.sdf_samples[:start]], 0)
            interval = np.random.choice(self.n_sdf_samples // self.num_samples - 2) + 1
            sampled = sdf_samples[::interval][:self.num_samples].astype(np.float32)
        else:
            sampled = self.sdf_samples[idx * self.num_samples: (idx + 1) * self.num_samples].astype(np.float32)
            # sampled = self.sdf_samples[:self.num_samples]
            # self.sdf_samples = self.sdf_samples[self.num_samples:]
        # sdfs = -sampled[..., 3:]
        # TODO: check sign
        sdfs = sampled[..., 3:]
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
# SDF dataset for 400M points
class SDF3Dataset(Dataset):
    v_center = 0
    v_scale = 0
    def __init__(self, path, size=100, num_samples=2**18, clip_sdf=None, part=0):
        super().__init__()
        self.path = path

        # load obj 
        # self.mesh = trimesh.load(path, force='mesh')
        # sdf_fname = self.config.data / "rand_surf-4m.ply"
        # if path is None:
            # path = self.path
        # else:
            # self.path = path
        path = f"{path}-{part}.ply"
        self.mesh = trimesh.load(path, process=False)
        sdf_onsurface = self.mesh.vertices
        #TODO: check bbox vs off_surface
        if True:
            # 4m-v1: with surface_offset
            # 4m-v0: without surface_offset
            # sdf_fname = f"{path.split('/rand')[0]}/near_surf-40m-v9.sdf"
            sdf_fname = path.replace("rand_surf", "near_surf")[:-3].replace("-cur1", "").replace("v9-", "exp5-") + "sdf"
            sdf_offsurface = read_sdf(sdf_fname)
            # sdf_offsurface = sdf_offsurface[:sdf_offsurface.shape[0]//3]
            # sdf_offsurface = sdf_offsurface[np.abs(sdf_offsurface[:, 3])<1]
            n_onsurface = sdf_onsurface.shape[0]
            n_offsurface = sdf_offsurface.shape[0]
            sdf_samples = np.zeros((n_onsurface + n_offsurface, 4))
            sdf_samples[:n_onsurface, :3] = sdf_onsurface
            sdf_samples[n_onsurface:] = sdf_offsurface

        # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
        # vs = self.mesh.vertices
        vs_full = sdf_samples[..., :3]
        vs = sdf_samples[:n_onsurface, :3]
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        # vs = (vs - v_center[None, :]) * v_scale
        if part == 0:
            SDF3Dataset.v_center = v_center
            SDF3Dataset.v_scale = v_scale
        else:
            v_center = SDF3Dataset.v_center
            v_scale = SDF3Dataset.v_scale
        vs_full = (vs_full - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        self.sdf_samples = sdf_samples
        # self.sdf_samples[..., :3] = vs
        self.sdf_samples[..., :3] = vs_full
        self.sdf_samples[..., 3] *= v_scale
        self.n_sdf_samples = sdf_samples.shape[0]

        choices = np.arange(self.n_sdf_samples)
        if size == 1:
            choices = choices[::100][:num_samples]
        else:
            np.random.shuffle(choices)
        self.sdf_samples = self.sdf_samples[choices]

        self.bounds_min = torch.FloatTensor(self.sdf_samples[..., :3].min(0))
        self.bounds_max = torch.FloatTensor(self.sdf_samples[..., :3].max(0))
        self.transform = np.eye(4)
        self.transform *= v_scale
        self.transform[:3, 3] = -v_center * v_scale
        self.transform[3, 3] = 1
        # print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        # if not self.mesh.is_watertight:
            # print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        # self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        self.num_samples = num_samples
        assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
        print(f"loaded {path}, {sdf_fname}")

    
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
        if False:
            start = np.random.choice(self.n_sdf_samples-1)
            sdf_samples = np.concatenate([self.sdf_samples[start:], self.sdf_samples[:start]], 0)
            interval = np.random.choice(self.n_sdf_samples // self.num_samples - 2) + 1
            sampled = sdf_samples[::interval][:self.num_samples].astype(np.float32)
        else:
            sampled = self.sdf_samples[idx * self.num_samples: (idx + 1) * self.num_samples].astype(np.float32)
        # sdfs = -sampled[..., 3:]
        # TODO: check sign
        sdfs = sampled[..., 3:]
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
        self.mesh = trimesh.load(path, process=False)
        sdf_onsurface = self.mesh.vertices
        if True:
            # sdf_fname = f"{path.split('/rand')[0]}/near_surf-200k.sdf"
            # 4m-v1: with surface_offset
            # 4m-v0: without surface_offset
            # sdf_fname = f"{path.split('/rand')[0]}/near_surf-40m-v9.sdf"
            sdf_fname = f"{path.split('/rand')[0]}/near_surf-40m-exp5.sdf"
            # faster read_sdf?
            sdf_offsurface = read_sdf(sdf_fname)
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
        vs_full = sdf_samples[..., :3]
        vs = sdf_samples[:n_onsurface, :3]
        vmin = vs.min(0)
        vmax = vs.max(0)
        v_center = (vmin + vmax) / 2
        v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
        # vs = (vs - v_center[None, :]) * v_scale
        vs_full = (vs_full - v_center[None, :]) * v_scale
        # self.mesh.vertices = vs
        self.sdf_samples = sdf_samples
        # pause()
        # self.sdf_samples[..., :3] = vs
        self.sdf_samples[..., :3] = vs_full
        self.sdf_samples[..., 3] *= v_scale
        self.n_sdf_samples = sdf_samples.shape[0]
        # pause()

        choices = np.arange(self.n_sdf_samples)
        np.random.shuffle(choices)
        self.sdf_samples = self.sdf_samples[choices]

        self.bounds_min = torch.FloatTensor(self.sdf_samples[..., :3].min(0))
        self.bounds_max = torch.FloatTensor(self.sdf_samples[..., :3].max(0))
        self.transform = np.eye(4)
        self.transform *= v_scale
        self.transform[:3, 3] = -v_center * v_scale
        self.transform[3, 3] = 1
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
        # pause()
        start = np.random.choice(self.n_sdf_samples-1)
        sdf_samples = np.concatenate([self.sdf_samples[start:], self.sdf_samples[:start]], 0)
        interval = np.random.choice(self.n_sdf_samples // self.num_samples - 2) + 1
        # interval = 1
        sampled = sdf_samples[::interval][:self.num_samples].astype(np.float32)
        # sdfs = -sampled[..., 3:]
        sdfs = sampled[..., 3:]
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
        # self.scale = v_scale
        # self.translation = v_center
        self.transform = np.eye(4)
        self.transform *= v_scale
        self.transform[:3, 3] = -v_center * v_scale
        self.transform[3, 3] = 1

        print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")

        if not self.mesh.is_watertight:
            print(f"[WARN] mesh is not watertight! SDF maybe incorrect.")
        #trimesh.Scene([self.mesh]).show()

        self.sdf_fn = pysdf.SDF(self.mesh.vertices, self.mesh.faces)
        
        # self.num_samples = num_samples * 4
        # assert self.num_samples % 8 == 0, "num_samples must be divisible by 8."
        self.clip_sdf = clip_sdf

        self.size = size
        self.bounds_min = torch.FloatTensor([-1, -1, -1])
        self.bounds_max = torch.FloatTensor([1, 1, 1])

    
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
            # points_uniform = np.random.rand(self.num_samples // 2, 3) * 2 - 1
            points_uniform = np.random.rand(self.num_samples // 2, 3)
            k = points_uniform.shape[0] // 2
            
            points_uniform[:k] = points_uniform[:k] * (self.mesh.bounds[1] - self.mesh.bounds[0])[None] + self.mesh.bounds[0][None]
            points_uniform[k:] = points_uniform[k:] * 2 - 1
            points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

            sdfs[self.num_samples // 4:] = -self.sdf_fn(points[self.num_samples // 4:])[:,None].astype(np.float32)
            # save created sdf samples
            if False:
                points_homo = np.ones((self.num_samples, 4))
                points_homo[:, :3] = points
                points_homo = points_homo@np.linalg.inv(self.transform).T 
                points_homo[:, 3:] = sdfs
                fname = "/home/fangyin/code/torch-ngp/t.sdf"
                write_sdf(fname, points_homo[self.num_samples//2:])
                pause()
 
        # clip sdf
        if self.clip_sdf is not None:
            sdfs = sdfs.clip(-self.clip_sdf, self.clip_sdf)

        results = {
            'sdfs': sdfs,
            'points': points,
        }

        #plot_pointcloud(points, sdfs)

        return results
