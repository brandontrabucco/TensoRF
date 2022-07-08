import torch
import torch.nn as nn

from torchvision.transforms import functional
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, Lambda

import clip

import os
import glob

from tqdm import tqdm
from collections import defaultdict

from .ray_utils import *
from .gpu_utils import get_rank, get_world_size


class THORDataset(object):

    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, 
                 scene_id=250, device="cuda"):

        self.rank = get_rank()
        self.world_size = get_world_size()

        self.scene_id = scene_id

        self.N_vis = N_vis
        self.root_dir = datadir

        self.split = split
        self.is_stack = is_stack
        
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

        self.scene_bbox = torch.tensor([[-20, -20, -20], [20, 20, 20]]).float()
        self.center = self.scene_bbox.mean(dim=0).view(1, 1, 3)
        self.radius = self.scene_bbox[1].view(1, 1, 3) - self.center

        self.convention = torch.tensor([[1, 0, 0, 0], 
                                        [0,-1, 0, 0], 
                                        [0, 0,-1, 0], 
                                        [0, 0, 0, 1]]).float()

        self.white_bg = True
        self.near_far = [0.05, 10.0]

        self.device = device

        self.model = clip.load("RN50x16", device=self.device)[0]
        self.model.visual.attnpool = nn.Identity()

        self.transform_clip = Compose([
            ToTensor(), Resize(self.model.visual.input_resolution), 
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                      (0.26862954, 0.2613026, 0.27577711))])

        self.transform_nerf = Compose([
            ToTensor(), Resize((self.img_wh[1], self.img_wh[0])),
            Lambda(lambda x: x.permute(1, 2, 0).view(np.prod(self.img_wh), 3))])

        self.prepare_dataset()

    def get_ray_features(self, image):

        with torch.no_grad():

            x = self.transform_clip(image).to(self.device)

            x = x.unsqueeze(0)
            x = self.model.visual(x).float()

            x = x.squeeze(0)
            x = x.permute(1, 2, 0)

            return x.cpu().view(np.prod(x.shape[:-1]), x.shape[-1])

    @property
    def all_rays(self):
        out = [v["rays"] for v in self.dataset_dict.values()]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)

    @property
    def all_rgbs(self):
        out = [v["image"] for v in self.dataset_dict.values()]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)

    @property
    def all_feat(self):
        out = [v["features"] for v in self.dataset_dict.values()]
        return torch.stack(out, dim=0) if self.is_stack else torch.cat(out, dim=0)
    
    def prepare_dataset(self):

        self.dataset_dict = defaultdict(dict)
        for file in list(glob.glob(os.path.join(self.root_dir, "*.npy"))):

            scene_id, transition_id, content = \
                os.path.basename(file)[:-4].split("-")[-3:]

            scene_id = int(scene_id)
            transition_id = int(transition_id)

            if scene_id == self.scene_id:
                self.dataset_dict[transition_id][content] = file

        num_obs = len(self.dataset_dict)
        dataset_obs = np.array(list(self.dataset_dict.keys()))

        np.random.shuffle(dataset_obs)

        train_obs = dataset_obs[:int(num_obs * 0.95)]
        val_obs = dataset_obs[int(num_obs * 0.95):]

        for key in (val_obs if self.split == "train" else train_obs):
            self.dataset_dict.pop(key)  # hide the remaining data split
                
        tqdm0 = tqdm if self.rank == 0 else lambda x: x
        for transition_id in tqdm0(list(self.dataset_dict.keys())):

            image = np.load(self.dataset_dict[transition_id]["image"])
            image = np.uint8(255. * image)  # convert to PIL image format

            self.dataset_dict[transition_id]["image"] = self.transform_nerf(image)
            self.dataset_dict[transition_id]["features"] = self.get_ray_features(image)

            pose = np.load(self.dataset_dict[transition_id]["pose"])
            self.dataset_dict[transition_id]["pose"] = pose.astype(np.float32)

            c2w = np.concatenate([np.roll(pose, -1, axis=-1), [[0, 0, 0, 1]]], axis=-2)
            c2w = torch.FloatTensor(c2w) @ self.convention

            self.dataset_dict[transition_id]["rays"] = \
                torch.cat(list(get_rays(self.directions, c2w)), dim=1).float()
    