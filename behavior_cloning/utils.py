import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from constants import TEXT_EMBEDDINGS


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


def get_embedding(task_name):
    """Returns embedding corresponding to the task_name"""
    if "open_drawer" in task_name:
        task_emb = TEXT_EMBEDDINGS[0]
    elif "close_drawer" in task_name:
        task_emb = TEXT_EMBEDDINGS[1]
    elif "pick_butter" in task_name:
        task_emb = TEXT_EMBEDDINGS[2]
    elif "place_butter" in task_name:
        task_emb = TEXT_EMBEDDINGS[3]
    elif "pick_toast" in task_name:
        task_emb = TEXT_EMBEDDINGS[4]
    elif "place_toast" in task_name:
        task_emb = TEXT_EMBEDDINGS[5]
    elif "cap_lid" in task_name:
        task_emb = TEXT_EMBEDDINGS[6]
    elif "pick_lid" in task_name:
        task_emb = TEXT_EMBEDDINGS[7]
    elif "pick_tea" in task_name:
        task_emb = TEXT_EMBEDDINGS[8]
    elif "place_lid" in task_name:
        task_emb = TEXT_EMBEDDINGS[9]
    elif "place_tea" in task_name:
        task_emb = TEXT_EMBEDDINGS[10]
    elif "uncap_lid" in task_name:
        task_emb = TEXT_EMBEDDINGS[11]
    elif "close_oven" in task_name:
        task_emb = TEXT_EMBEDDINGS[12]
    elif "open_oven" in task_name:
        task_emb = TEXT_EMBEDDINGS[13]
    elif "place_bowl" in task_name:
        task_emb = TEXT_EMBEDDINGS[14]
    elif "slide_out" in task_name:
        task_emb = TEXT_EMBEDDINGS[15]
    elif "cap_mug" in task_name:
        task_emb = TEXT_EMBEDDINGS[16]
    elif "pick_mug" in task_name:
        task_emb = TEXT_EMBEDDINGS[17]
    elif "pick_towel" in task_name:
        task_emb = TEXT_EMBEDDINGS[18]
    elif "wipe_towel" in task_name:
        task_emb = TEXT_EMBEDDINGS[19]
    elif "pick_cup" in task_name:
        task_emb = TEXT_EMBEDDINGS[20]
    elif "place_cup" in task_name:
        task_emb = TEXT_EMBEDDINGS[21]
    elif "open_box" in task_name:
        task_emb = TEXT_EMBEDDINGS[38]
    else:
        task_emb = TEXT_EMBEDDINGS[0]
    return task_emb
