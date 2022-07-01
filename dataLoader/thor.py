import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional

import os
import json
import glob

import itertools
from tqdm import tqdm
from collections import OrderedDict

from .ray_utils import *
from .gpu_utils import get_rank, get_world_size


class THORDataset(Dataset):

    def __init__(self, datadir, split='train', downsample=3.5, is_stack=False, N_vis=-1, 
                 num_chunks=128, images_per_scene=199):

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

        self.scene_bbox = torch.tensor([[-20, -20, -20], [20, 20, 20]]).float()
        self.convention = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).float()

        self.prepare()

        self.white_bg = True
        self.near_far = [0.0, 10.0]
        
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1].view(1, 1, 3) - self.center).float()

    @property
    def all_rays(self):
        out = [vi["rays"] for v in self.dataset_dict.values() for vi in v]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)

    @property
    def all_rgbs(self):
        out = [self.process_image(np.load(vi["image"])) for v in self.dataset_dict.values() for vi in v]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)

    @property
    def all_clip(self):
        out = [torch.FloatTensor(vi["clip"]) for v in self.dataset_dict.values() for vi in v]
        return torch.stack(out, dim=0)
    
    def prepare(self):

        w, h = self.img_wh

        self.directions = get_ray_directions(h, w, [self.focal, self.focal])
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        self.intrinsics = torch.tensor([[self.focal, 0, w / 2],
                                        [0, self.focal, h / 2], 
                                        [0, 0, 1]]).float()

        self.dataset_dict = OrderedDict()

        for file in list(glob.glob(os.path.join(self.root_dir, "*.npy"))):

            scene_id, transition_id, content = \
                os.path.basename(file)[:-4].split("-")[-3:]

            scene_id = int(scene_id)
            transition_id = int(transition_id)

            if scene_id not in self.dataset_dict:
                self.dataset_dict[scene_id] = [
                    dict() for _ in range(self.images_per_scene)]

            self.dataset_dict[scene_id][transition_id][content] = file

        self.num_scenes = len(self.dataset_dict)
        self.scenes = np.array(list(self.dataset_dict.keys()))

        self.train_scenes = self.scenes[:int(self.num_scenes * 0.99)]
        self.val_scenes = self.scenes[int(self.num_scenes * 0.99):]

        self.train_chunk = np.array_split(
            self.train_scenes, self.world_size)[self.rank]
            
        self.val_chunk = np.array_split(
            self.val_scenes, self.world_size)[self.rank]

        self.chunk = self.train_chunk if self.split == "train" else self.val_chunk
        self.chunk = np.array([0, 500, 1000, 1500, 2000, 2500])

        for scene_id in self.scenes:
            if scene_id not in self.chunk:
                self.dataset_dict.pop(scene_id)
                
        tqdm0 = tqdm if self.rank == 0 else lambda x: x

        for scene_id in tqdm0(list(self.dataset_dict.keys())):

            canonical = None  # coordinates are relative to first observation

            for transition_id in range(self.images_per_scene):

                clip = np.load(self.dataset_dict[scene_id][transition_id]["clip"])
                self.dataset_dict[scene_id][transition_id]["clip"] = clip.astype(np.float32)

                pose = np.load(self.dataset_dict[scene_id][transition_id]["pose"])
                self.dataset_dict[scene_id][transition_id]["pose"] = pose.astype(np.float32)

                c2w = torch.FloatTensor(np.concatenate([
                    np.roll(pose, -1, axis=-1), [[0, 0, 0, 1]]], axis=-2)) @ self.convention

                if canonical is None:
                    canonical = torch.linalg.inv(c2w)

                c2w = canonical @ c2w

                self.dataset_dict[scene_id][transition_id]["rays"] = \
                    torch.cat(list(get_rays(self.directions, c2w)), dim=1).float()
        
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

    def process_image(self, image):

        image = torch.as_tensor(image, dtype=torch.float32)

        image = image.permute(2, 0, 1)
        image = functional.resize(image, self.img_wh)
        image = image.permute(1, 2, 0)

        return image.view(np.prod(self.img_wh), 3)

    def __getitem__(self, idx):

        chunk_idx, view_idx, scene_idx = self.idx2sample(idx)
        data = {k: np.load(v) if isinstance(v, str) else v 
                for k, v in self.dataset_dict[scene_idx][view_idx].items()}

        image = self.process_image(data["image"])

        size = np.prod(self.img_wh) // self.num_chunks
        indices = slice(size * chunk_idx, size * (chunk_idx + 1))

        return {"rays": data["rays"][indices], "rgbs": image[indices]}
