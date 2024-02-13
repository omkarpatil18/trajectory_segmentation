import numpy as np
import torch
import cv2
import os
import re
import matplotlib.pyplot as plt
from einops import rearrange
from policy import ACTPolicy, CNNMLPPolicy
from dataset.task_constants import SIM_TASK_CONFIG


### Functions used in training and testing

FRANKA_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    ],
    dtype=np.float32,
).T


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs, camera_names):
    """Returns image in BGR format."""
    curr_images = []
    viz_out = {}
    for cam_name in camera_names:
        curr_image = getattr(obs, cam_name)
        viz_out[cam_name] = curr_image
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR)
        curr_image = rearrange(curr_image, "h w c -> c h w")
        curr_images.append(curr_image)

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image, viz_out


def get_embedding(task_name, task_desc):
    """Returns embedding corresponding to the task_desc"""
    center_place_re = re.compile(r"(Place the )([\w ]*)(block at the green center.)")
    block_place_re = re.compile(
        r"(Place the )([\w ]*)(block on top of the [\w]* block.)"
    )
    task_desc = re.sub(center_place_re, r"\1\3", task_desc)
    task_desc = re.sub(block_place_re, r"\1\3", task_desc)

    skill_emb_map = SIM_TASK_CONFIG[task_name]["skill_emb"]
    if skill_emb_map is None or task_desc == "":
        return [0] * 384  # return zeros if emb is not found
    return skill_emb_map[task_desc]


def save_videos(video, dt=0.02, video_path=None):
    """
    Save videos of rollouts
    video: list
        Torch tensor of images [num_cam, depth, height, width]
    """
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f"Saved video to: {video_path}")
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f"Saved video to: {video_path}")


### helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_images_from_folder(folder):
    "Load all the images in a folder sorted by name"
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_image_from_folder(folder, ts):
    """Loads images in BGR mode"""
    "Load the image corresponding to a given timestep from a folder"
    img = cv2.imread(os.path.join(folder, f"{ts}.png"))
    if img is None:
        raise Exception(f"Image corresponding to index {ts} not found in {folder}")
    return img


##### Dataset management ######
def merge_data():
    import shutil

    for i in range(250):
        shutil.move(
            f"/home/local/ASUAD/opatil3/datasets/shoes_in_box_temporal_3/s_put_shoes_in_box/variation0/episodes/episode{i}",
            f"/home/local/ASUAD/opatil3/datasets/shoes_in_box_temporal_1/s_put_shoes_in_box/variation0/episodes/episode{i+500}",
        )


# merge_data()
