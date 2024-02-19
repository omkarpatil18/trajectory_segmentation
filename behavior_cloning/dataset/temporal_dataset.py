import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import sys
import os
import random
import re
from functools import partial
from dataset.pickle_dataset import ConfigMinMaxNormalization
import cv2

sys.path.append("/home/local/ASUAD/opatil3/src/trajectory_segmentation/")
from utils import get_embedding, load_image_from_folder
from constants import CAMERA_NAMES
from dataset.task_constants import SIM_TASK_CONFIG, IMG_SIZE

CONFIG_DIM = 7  # joint space


def load_temporal_data(
    skill_or_task="skill_or_task",
    data_dir="/home/local/ASUAD/opatil3/datasets/temporal",
    chunk_size=100,
    norm_bound=None,
    batch_size=8,
    train_split=0.8,
    num_datapoints=750,
):

    file_list = []
    task_re = SIM_TASK_CONFIG[skill_or_task]["train_subtasks"][0]
    regex = re.compile(task_re)
    for dirs in os.listdir(data_dir):
        if regex.search(dirs):
            search_dir = os.path.join(data_dir, dirs, "variation0", "episodes")
            for dir in os.listdir(search_dir):
                file_list.append((skill_or_task, os.path.join(search_dir, dir)))

    random.shuffle(file_list)
    file_list = file_list[:int(num_datapoints)]

    split_idx = int(len(file_list) * train_split)
    train_dataset = RLBenchTemporalDataset(
        file_list=file_list[:split_idx],
        chunk_size=chunk_size,
        norm_bound=norm_bound,
    )
    val_dataset = RLBenchTemporalDataset(
        file_list=file_list[split_idx:],
        chunk_size=chunk_size,
        norm_bound=norm_bound,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_loader, val_loader


class RLBenchTemporalDataset(Dataset):
    """Dataset for temporally segmented trajectories for BC"""

    def __init__(
        self,
        file_list,
        chunk_size=100,
        norm_bound=None,
        sampler=partial(np.random.uniform, 0, 1),  # partial function
    ):
        self.file_list = file_list
        self.chunk_size = chunk_size
        if norm_bound is not None:
            self.normalizer = ConfigMinMaxNormalization(norm_bound)
        else:
            self.normalizer = None
        self.len = len(self.file_list)
        self.sampler = sampler

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # TODO: MT-ACT starts action from start_ts -1!
        task_name, data_dir = self.file_list[index]
        image_dict = {}
        data_batch = {}

        # Read low dim data from the folders
        gripper_open = []
        joint_positions = []
        # gripper_pose = []
        # gripper_joint_positions = []

        with open(os.path.join(data_dir, "low_dim_obs.pkl"), "rb") as f:
            demo_obj = pickle.load(f)
            obs_list = demo_obj._observations
            task_desc = demo_obj.instructions[0]
            for obs in obs_list:
                gripper_open.append(obs.gripper_open)
                joint_positions.append(obs.joint_positions)
                # gripper_pose.append(obs.gripper_pose)
                # gripper_joint_positions.append(obs.gripper_joint_positions)

        episode_len = len(joint_positions)
        start_ts = int(self.sampler() * episode_len)
        end_ts = min(episode_len, start_ts + self.chunk_size)

        # Process joint action
        chunk_data = joint_positions[start_ts:end_ts]
        if self.normalizer is not None:
            for i, js in enumerate(chunk_data):
                if self.normalizer.validate_bounds(js):
                    chunk_data[i] = self.normalizer.transform(js)
                else:
                    chunk_data[i] = self.normalizer.clamp(js)

        data = torch.zeros((self.chunk_size, CONFIG_DIM))
        data[: end_ts - start_ts, :] = torch.as_tensor(np.array(chunk_data))
        data_batch["joint_action"] = data

        # Process discrete gripper action
        chunk_data = gripper_open[start_ts:end_ts]
        data = torch.zeros((self.chunk_size))
        data[: end_ts - start_ts] = torch.as_tensor(np.array(chunk_data))
        data_batch["gripper_action"] = data

        # Process images
        rgb_regex = re.compile(r"rgb")
        for dir in os.listdir(data_dir):
            if rgb_regex.search(dir):
                img = load_image_from_folder(os.path.join(data_dir, dir), ts=start_ts)
                image_dict[dir] = img

        all_cam_images = []
        for cam_name in CAMERA_NAMES:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = torch.from_numpy(np.stack(all_cam_images, axis=0))
        image_data = torch.einsum("k h w c -> k c h w", all_cam_images)  # channel last
        image_data = image_data / 255.0  # normalize image and change dtype to float
        data_batch["images"] = image_data

        # Process padding
        is_pad = np.ones((self.chunk_size))
        is_pad[: end_ts - start_ts] = 0
        data_batch["is_pad"] = torch.from_numpy(is_pad).bool()

        # Process task_embeddings
        task_emb = get_embedding(task_name, task_desc=task_desc[0])
        data_batch["task_emb"] = torch.as_tensor(task_emb, dtype=torch.float)

        # Verify dimensions
        assert data_batch["images"].shape == torch.Size(
            [len(CAMERA_NAMES), 3, *IMG_SIZE]
        )
        assert data_batch["is_pad"].shape == torch.Size([self.chunk_size])
        assert data_batch["joint_action"].shape == torch.Size([self.chunk_size, 7])
        assert data_batch["gripper_action"].shape == torch.Size([self.chunk_size])
        return data_batch
