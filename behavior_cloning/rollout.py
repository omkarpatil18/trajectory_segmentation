import torch
import cv2
import re
import numpy as np
import os, sys
import datetime

sys.path.append(
    "/home/local/ASUAD/opatil3/src/trajectory_segmentation/behavior_cloning/"
)
from constants import CAMERA_NAMES
from utils import (
    set_seed,
    save_videos,
)  # helper functions
from utils import get_embedding, make_policy, FRANKA_JOINT_LIMITS, get_image
from dataset.task_constants import SIM_TASK_CONFIG, IMG_SIZE, model_path_dict

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


def eval_bc(config, ckpt_name, save_episode=True, **kwargs):
    set_seed(42)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    temporal_agg = config["temporal_agg"]
    rlbench_env = config["rlbench_env"]
    task_name = config["task_name"]
    seq_skills = config["seq_skills"]
    num_datapoints = config["num_datapoints"]

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    skill_or_task_models = {}
    if seq_skills:  # load all the skill models to sequence
        skill_sequence = SIM_TASK_CONFIG[task_name]["lang_to_skill_map"].values()
        for skill in skill_sequence:
            policy_path = os.path.join(model_path_dict[skill], ckpt_name)
            policy_path = policy_path.replace("xxx", str(num_datapoints))
            policy = make_policy(policy_class, policy_config)
            loading_status = policy.load_state_dict(torch.load(policy_path))
            print(loading_status)
            policy.cuda().eval()
            skill_or_task_models[skill] = policy
            print(f"Loaded: {policy_path}")
    else:  # load the single model to execute
        policy = make_policy(policy_class, policy_config)
        loading_status = policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        policy.cuda().eval()
        skill_or_task_models[task_name] = policy
        print(f"Loaded: {ckpt_path}")

    #  A simple MinMax transformation
    min_bound = FRANKA_JOINT_LIMITS[:, 0]
    max_bound = FRANKA_JOINT_LIMITS[:, 1]
    pre_process = (
        lambda s_pos: (1.0 * (s_pos - min_bound) / (max_bound - min_bound)) * 2.0 - 1
    )
    post_process = (
        lambda s_pos: 1.0 * ((s_pos + 1) / 2) * (max_bound - min_bound) + min_bound
    )

    # Training config
    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]
    num_rollouts = 50

    # load simulation environment
    obs_config = ObservationConfig()
    obs_config.right_shoulder_camera.image_size = IMG_SIZE
    obs_config.left_shoulder_camera.image_size = IMG_SIZE
    obs_config.overhead_camera.image_size = IMG_SIZE
    obs_config.wrist_camera.image_size = IMG_SIZE
    obs_config.front_camera.image_size = IMG_SIZE
    obs_config.set_all(True)
    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
        ),
        obs_config=obs_config,
        robot_setup="panda",
        headless=not onscreen_render,
    )
    env.launch()

    # Iterate over tasks
    task_performances = {}
    task = env.get_task(rlbench_env)
    task_dir = os.path.join(
        ckpt_dir, rlbench_env.__name__, f"{datetime.datetime.now()}"
    )  # Store task specific information here
    if not os.path.isdir(task_dir):
        os.makedirs(task_dir)
    task_status = []

    for rollout_id in range(num_rollouts):
        _, obs = task.reset()
        image_list = []  # for visualization

        # Get text embeddding based on seq_skills
        if seq_skills:
            task_desc = obs.instruction[0]
        else:  # single model rollout, not multi_task
            task_desc = [""]
        with torch.inference_mode():
            for task_idx, task_d in enumerate(task_desc):
                # Choose the model to rollout
                if seq_skills:
                    for k, v in SIM_TASK_CONFIG[task_name]["lang_to_skill_map"].items():
                        if re.search(k, task_d):
                            rollout_skill_or_task = v
                else:
                    rollout_skill_or_task = task_name
                max_timesteps = SIM_TASK_CONFIG[rollout_skill_or_task]["episode_len"]
                rollout_model = skill_or_task_models[rollout_skill_or_task]

                # Get embedding for task description
                task_emb = torch.as_tensor(
                    [get_embedding(rollout_skill_or_task, task_d)], dtype=torch.float
                ).cuda()  # get text embedding for the task

                # Define variables required for rollout
                if temporal_agg:
                    all_time_actions = torch.zeros(
                        [max_timesteps, max_timesteps + num_queries, state_dim]
                    ).cuda()

                for t in range(max_timesteps):
                    joint_position = pre_process(obs.joint_positions)
                    qpos_numpy = np.array(np.hstack([joint_position, obs.gripper_open]))
                    qpos = torch.from_numpy(qpos_numpy).float().cuda().unsqueeze(0)
                    curr_image, viz_out = get_image(obs, camera_names)
                    image_list.append(viz_out)  # For generating videos

                    ### query policy
                    if t % query_frequency == 0:
                        all_actions = rollout_model(
                            qpos,
                            curr_image,
                            task_emb=task_emb,
                        )
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(
                            actions_for_curr_step != 0, axis=1
                        )
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = (
                            torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        )
                        raw_action = (actions_for_curr_step * exp_weights).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        raw_action = all_actions[:, t % query_frequency]

                    # post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = np.hstack([post_process(raw_action[:7]), raw_action[-1]])
                    target_qpos = action

                    # step the environment
                    obs, reward, terminate = task.step(target_qpos)

                    # check for success condition
                    if obs.success_state[task_idx] and seq_skills:
                        break  # break out of the fixed timestep rollout loop

        task_status.append(obs.success_state)
        print(f"Rollout for {rlbench_env}: {rollout_id}")
        if save_episode:
            save_videos(
                image_list,
                video_path=os.path.join(
                    task_dir,
                    f"video{rollout_id}-{task_status[rollout_id] == [True]*len(task_status[rollout_id])}.mp4",
                ),
            )

    avg_per_task_status = np.mean(task_status, axis=0)
    task_success_rate = np.mean(
        [(1 if task_s == [True] * len(task_s) else 0) for task_s in task_status]
    )
    # save success rate to txt
    result_file_name = "result_" + ckpt_name.split(".")[0] + f".txt"
    with open(os.path.join(task_dir, result_file_name), "w") as f:
        f.write(str(config))
        f.write("\n\n")
        f.write(f"Average per-task status: {str(avg_per_task_status)}")
        f.write("\n\n")
        f.write(f"Task_success_rate: {str(task_success_rate)}")
        f.write("\n\n")
        f.write(
            "\n".join(
                [
                    f"{epi_num} : {t_status}"
                    for epi_num, t_status in enumerate(task_status)
                ]
            )
        )

    task_performances[rlbench_env.__name__] = [avg_per_task_status]
    env.shutdown()
    return task_performances
