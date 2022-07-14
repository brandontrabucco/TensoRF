import torch
import torch.nn as nn

from torchvision.transforms import functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda

import clip

import os
import glob

from tqdm import tqdm
from collections import defaultdict

from .ray_utils import *


class THORDataset(object):

    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, scene_id=2000, phase="walkthrough", stage="train"):

        self.N_vis = N_vis
        self.root_dir = datadir

        self.split = split
        self.is_stack = is_stack

        self.scene_id = scene_id
        self.phase = phase
        self.stage = stage
        
        self.img_wh = [int(224 / downsample), int(224 / downsample)]

        self.focal = 0.5 * self.img_wh[0] / np.tan(0.25 * np.pi)

        self.intrinsics = torch.tensor([[self.focal, 0, self.img_wh[0] / 2],
                                        [0, self.focal, self.img_wh[1] / 2], 
                                        [0, 0, 1]]).float()

        self.directions = get_ray_directions(self.img_wh[1], 
                                             self.img_wh[0], 
                                             [self.focal, self.focal])

        self.directions = self.directions / \
            torch.norm(self.directions, dim=-1, keepdim=True)

        self.convention = torch.tensor([[1, 0, 0, 0], 
                                        [0,-1, 0, 0], 
                                        [0, 0,-1, 0], 
                                        [0, 0, 0, 1]]).float()

        self.scene_bbox = torch.tensor(np.load(os.path.join(
            datadir, f"thor-{phase}-{stage}-{scene_id}-bounds.npy"))).float()

        self.center = self.scene_bbox.mean(dim=0).view(1, 1, 3)
        self.radius = self.scene_bbox[1].view(1, 1, 3) - self.center

        self.white_bg = False
        self.near_far = [0.05, torch.linalg.norm(
            self.scene_bbox[1] - self.scene_bbox[0]).cpu().numpy().item()]

        self.transform_nerf = Compose([
            ToTensor(), Resize((self.img_wh[1], self.img_wh[0])),
            Lambda(lambda x: x.permute(1, 2, 0).view(np.prod(self.img_wh), x.shape[0]))])

        self.prepare_dataset()  # load images and segmentation into memory

    @property
    def all_rays(self):
        out = [v["rays"] for v in self.dataset_dict.values()]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)

    @property
    def all_rgbs(self):
        out = [v["image"] for v in self.dataset_dict.values()]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)

    @property
    def all_labels(self):
        out = [v["segmentation"] for v in self.dataset_dict.values()]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)
    
    def prepare_dataset(self):

        self.dataset_dict = defaultdict(dict)

        for file in list(glob.glob(os.path.join(
                self.root_dir, f"thor-{self.phase}-{self.stage}-{self.scene_id}-*.npy"))):

            if "bounds" in file:
                continue  # skip the scene bounds

            scene_id, transition_id, content = \
                os.path.basename(file)[:-4].split("-")[-3:]

            scene_id = int(scene_id)
            transition_id = int(transition_id)

            self.dataset_dict[transition_id][content] = file

        num_obs = len(self.dataset_dict)
        dataset_obs = np.array(list(self.dataset_dict.keys()))

        dataset_obs.sort()

        train_obs = dataset_obs[:int(num_obs * 0.95)]
        val_obs = dataset_obs[int(num_obs * 0.95):]

        for key in (val_obs if self.split == "train" else train_obs):
            self.dataset_dict.pop(key)  # hide samples in the remaining split
                
        for transition_id in tqdm(list(self.dataset_dict.keys())):

            image = np.load(self.dataset_dict[transition_id]["image"])
            self.dataset_dict[transition_id]["image"] = self.transform_nerf(image)

            segmentation = np.load(self.dataset_dict[transition_id]["segmentation"])
            segmentation = torch.LongTensor(segmentation.astype(np.int64)).permute(2, 0, 1)
            segmentation = functional.resize(segmentation, (self.img_wh[1], self.img_wh[0]), 
                                             interpolation=InterpolationMode.NEAREST)
            self.dataset_dict[transition_id]["segmentation"] = segmentation.view(np.prod(self.img_wh))

            pose = np.load(self.dataset_dict[transition_id]["pose"])
            self.dataset_dict[transition_id]["pose"] = pose.astype(np.float32)

            c2w = torch.FloatTensor(pose) @ self.convention

            self.dataset_dict[transition_id]["rays"] = \
                torch.cat(list(get_rays(self.directions, c2w)), dim=1).float()
    
