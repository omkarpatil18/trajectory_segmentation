import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import pickle
import time
import lzma
import os
from task_constants import SIM_TASK_CONFIG


class TrajGen:
    def __init__(self, robot="panda", headless=True):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        self.env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
            ),
            obs_config=ObservationConfig(),
            robot_setup=robot,
            headless=headless,
        )
        self.env.launch()

    def get_demo(self, task_name):
        Task = SIM_TASK_CONFIG[task_name]["rlbench_env"]
        demo_task = self.env.get_task(Task)
        print(demo_task.get_name())
        demo_task.sample_variation()
        start = time.perf_counter()
        demo = demo_task.get_demos(1, live_demos=True)[0]
        demo_dict = dict(
            front_rgb=[],
            front_mask=[],
            front_depth=[],
            front_point_cloud=[],
            left_shoulder_rgb=[],
            left_shoulder_mask=[],
            left_shoulder_depth=[],
            left_shoulder_point_cloud=[],
            right_shoulder_rgb=[],
            right_shoulder_mask=[],
            right_shoulder_depth=[],
            right_shoulder_point_cloud=[],
            overhead_rgb=[],
            overhead_mask=[],
            overhead_depth=[],
            overhead_point_cloud=[],
            wrist_rgb=[],
            wrist_mask=[],
            wrist_depth=[],
            wrist_point_cloud=[],
            joint_positions=[],
            joint_velocities=[],
            gripper_pose=[],
            gripper_open=[],
        )
        demo_dict["task_name"] = task_name
        for obs in demo._observations:
            demo_dict["front_rgb"].append(obs.front_rgb)
            demo_dict["front_mask"].append(obs.front_mask)
            demo_dict["front_depth"].append(obs.front_depth)
            # demo_dict['front_point_cloud'].append(obs.front_point_cloud)
            demo_dict["left_shoulder_rgb"].append(obs.left_shoulder_rgb)
            demo_dict["left_shoulder_mask"].append(obs.left_shoulder_mask)
            demo_dict["left_shoulder_depth"].append(obs.left_shoulder_depth)
            # demo_dict['left_shoulder_point_cloud'].append(obs.left_shoulder_point_cloud)
            demo_dict["right_shoulder_rgb"].append(obs.right_shoulder_rgb)
            demo_dict["right_shoulder_mask"].append(obs.right_shoulder_mask)
            demo_dict["right_shoulder_depth"].append(obs.right_shoulder_depth)
            # demo_dict['right_shoulder_point_cloud'].append(obs.right_shoulder_point_cloud)
            demo_dict["overhead_rgb"].append(obs.overhead_rgb)
            demo_dict["overhead_mask"].append(obs.overhead_mask)
            demo_dict["overhead_depth"].append(obs.overhead_depth)
            # demo_dict['overhead_point_cloud'].append(obs.overhead_point_cloud)
            demo_dict["wrist_rgb"].append(obs.wrist_rgb)
            demo_dict["wrist_mask"].append(obs.wrist_mask)
            demo_dict["wrist_depth"].append(obs.wrist_depth)
            # demo_dict['wrist_point_cloud'].append(obs.wrist_point_cloud)
            demo_dict["joint_positions"].append(obs.joint_positions)
            demo_dict["joint_velocities"].append(obs.joint_velocities)
            demo_dict["gripper_pose"].append(obs.gripper_pose)
            demo_dict["gripper_open"].append(obs.gripper_open)

        print(f"demo generated in: {round(time.perf_counter()-start,3)}s")
        return demo_dict

    def generate_data(self, task_name, dataset_path, num_demos_per_task=100):
        dataset_path = os.path.join(dataset_path, task_name)
        if not os.path.isdir(dataset_path):
            os.makedirs(dataset_path)
        for j in range(num_demos_per_task):
            with open(
                os.path.join(
                    dataset_path,
                    f"demo_{j}.pickle",
                ),
                "wb",
            ) as f:
                pickle.dump(self.get_demo(task_name), f)


traj_gen = TrajGen(headless=True)
traj_gen.generate_data(
    task_name="sim_open_box",
    dataset_path="/home/local/ASUAD/opatil3/datasets/rlbench",
    num_demos_per_task=100,
)
