import argparse, sys, os

sys.path.append(
    "/home/local/ASUAD/opatil3/src/trajectory_segmentation/behavior_cloning/"
)

from behavior_cloning.dataset.pickle_dataset import load_data
from behavior_cloning.dataset.temporal_dataset import load_temporal_data
import numpy as np
from task_constants import DATA_DIR


FRANKA_JOINT_LIMITS = np.asarray(
    [
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    ],
    dtype=np.float32,
).T


def main():
    # train_loader, val_loader = load_data(
    #     task_name="sim_open_drawer",
    #     required_data_keys=[
    #         "overhead_rgb",
    #         "left_shoulder_rgb",
    #         "right_shoulder_rgb",
    #         "wrist_rgb",
    #         "joint_positions",
    #         "gripper_open",
    #     ],
    #     chunk_size=20,
    #     norm_bound=None,
    #     batch_size=8,
    #     train_split=0.8,
    # )

    train_loader, val_loader = load_temporal_data(
        skill_or_task="sim_skill_pick",
        data_dir="/home/local/ASUAD/opatil3/datasets/temporal/task_data",
        chunk_size=100,
        norm_bound=None,
        batch_size=8,
        train_split=0.8,
    )

    for train_data in train_loader:
        break


main()
