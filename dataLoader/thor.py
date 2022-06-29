import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional

import os
import json
import glob

import itertools
from tqdm import tqdm

from .ray_utils import *
from .gpu_utils import get_rank, get_world_size


class THORDataset(Dataset):

    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, 
                 num_chunks=64, images_per_scene=199):

        self.rank = get_rank()
        self.world_size = get_world_size()

        self.num_chunks = num_chunks
        self.images_per_scene = images_per_scene

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        
        self.img_wh = [int(224 / downsample), int(224 / downsample)]
        self.focal = 0.5 * self.img_wh[0] / np.tan(0.25 * np.pi)

        self.scene_bbox = torch.tensor([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]])
        self.thor2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.prepare()

        self.white_bg = True
        self.near_far = [0.01, 10.0]
        
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1].view(1, 1, 3) - self.center).float()

        self.all_rgbs = []
        self.all_rays = []

        for sample_idx in range(len(self)):

            sample = self[sample_idx]

            self.all_rgbs.append(sample["rgbs"])
            self.all_rays.append(sample["rays"])

        self.all_rays = torch.cat(self.all_rays, dim=0)
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)

        if is_stack:

            self.all_rays = self.all_rays.reshape(
                -1, self.img_wh[1]* self.img_wh[0], 6)

            self.all_rgbs = self.all_rgbs.reshape(
                -1, self.img_wh[1], self.img_wh[0], 3)
    
    def prepare(self):

        w, h = self.img_wh

        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal, 0, w/2],[0, self.focal, h/2],[0, 0, 1]]).float()

        self.dataset_dict = dict()

        for file in list(glob.glob(os.path.join(self.root_dir, "*.npy"))):

            scene_id, transition_id, content = \
                os.path.basename(file)[:-4].split("-")[-3:]

            scene_id = int(scene_id)
            transition_id = int(transition_id)

            if scene_id not in self.dataset_dict:
                self.dataset_dict[scene_id] = [dict() for _ in range(self.images_per_scene)]

            self.dataset_dict[scene_id][transition_id][content] = file

        self.num_scenes = len(self.dataset_dict)
        self.scenes = np.array(list(self.dataset_dict.keys()))

        self.train_scenes = self.scenes[:int(self.num_scenes * 0.95)]
        self.val_scenes = self.scenes[int(self.num_scenes * 0.95):]

        self.train_chunk = np.array_split(
            self.train_scenes, self.world_size)[self.rank]
            
        self.val_chunk = np.array_split(
            self.val_scenes, self.world_size)[self.rank]

        self.chunk = np.array([2550])

        for scene_id in self.scenes:
            if scene_id not in self.chunk:
                self.dataset_dict.pop(scene_id)
                
        tqdm0 = tqdm if self.rank == 0 else lambda x: x

        for scene_id, transition_id in tqdm0(list(
                itertools.product(self.dataset_dict.keys(), 
                                  range(self.images_per_scene)))):

                self.dataset_dict[scene_id][transition_id]["clip"] = \
                    np.load(self.dataset_dict[scene_id][transition_id]["clip"])

                self.dataset_dict[scene_id][transition_id]["pose"] = \
                    np.load(self.dataset_dict[scene_id][transition_id]["pose"])
        
    def __len__(self):

        return (self.num_chunks * 
                self.images_per_scene * self.chunk.size)

    def idx2sample(self, idx):

        chunk_idx = idx % self.num_chunks
        idx = idx // self.num_chunks

        target_view_idx = idx % self.images_per_scene
        idx = idx // self.images_per_scene

        return (chunk_idx, 
                target_view_idx, self.chunk[idx])

    def __getitem__(self, idx):

        chunk_idx, view_idx, scene_idx = self.idx2sample(idx)

        data = {k: np.load(v) if isinstance(v, str) else v 
                for k, v in self.dataset_dict[scene_idx][view_idx].items()}

        image = data["image"]
        pose = data["pose"]

        image = torch.FloatTensor(image).permute(2, 0, 1)
        image = functional.resize(image, self.img_wh).permute(1, 2, 0)
        image = image.view(np.prod(self.img_wh), 3)

        c2w = np.roll(pose, shift=-1, axis=-1)
        c2w = np.concatenate([c2w, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)
        c2w = torch.FloatTensor(c2w @ self.thor2opencv)

        rays_o, rays_d = get_rays(self.directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)

        chunk_size = image.shape[0] // self.num_chunks
        indices = slice(chunk_size * chunk_idx, chunk_size * (chunk_idx + 1))

        return {"rays": rays[indices], "rgbs": image[indices]}
