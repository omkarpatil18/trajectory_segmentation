import torch
import numpy as np
import os, sys
import json
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

sys.path.append(
    "/home/local/ASUAD/opatil3/src/trajectory_segmentation/behavior_cloning/"
)
from constants import CAMERA_NAMES
from utils import (
    compute_dict_mean,
    set_seed,
    make_policy,
    make_optimizer,
    detach_dict,
    FRANKA_JOINT_LIMITS,
)  # helper functions
from dataset.pickle_dataset import load_data as load_rlbench_data
from dataset.temporal_dataset import load_temporal_data
from rollout import eval_bc


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args["eval"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size = args["batch_size"]
    num_epochs = args["num_epochs"]
    ckpt_dir = args["ckpt_dir"]
    ckpt_names = args["ckpt_names"]
    data_dir = args["data_dir"]
    transformer_only = args["transformer_only"]

    # get task parameters
    is_sim = task_name[:4] == "sim_"
    from dataset.task_constants import SIM_TASK_CONFIG, DATA_DIR

    task_config = SIM_TASK_CONFIG[task_name]
    rlbench_env = task_config["rlbench_env"]

    # fixed parameters
    state_dim = 8
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": CAMERA_NAMES,
            "state_dim": state_dim,
            "transformer_only": transformer_only,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": CAMERA_NAMES,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,  # dir to store model checkpoints during training and testing
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": CAMERA_NAMES,
        "real_robot": not is_sim,
        "rlbench_env": rlbench_env,
        "multi_task": args["multi_task"],  # decides whether lang emb should be used
        "batch_size": args["batch_size"],
        "seq_skills": args[
            "seq_skills"
        ],  # whether multiple models need to be sequenced during rollout
        "model_path_dict": args[
            "model_path_dict"
        ],  # dict of model path to be sequenced together
    }
    if is_eval:
        if len(ckpt_names) == 0:
            ckpt_names = [f"policy_best.ckpt"]
        print(f"Evaluating for {ckpt_names}")
        results = []
        for ckpt_name in ckpt_names:
            task_perfs = eval_bc(config, ckpt_name, save_episode=True)
            for key in task_perfs.keys():
                results.append([key, task_perfs[key]])

        for ckpt_name, [success_rate, avg_return] in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        print()
        exit()

    # required_data_keys = [
    #     "overhead_rgb",
    #     "left_shoulder_rgb",
    #     "right_shoulder_rgb",
    #     "wrist_rgb",
    #     "front_rgb",
    #     "joint_positions",
    #     "gripper_open",
    # ]
    # train_dataloader, val_dataloader = load_rlbench_data(
    #     task_name=args["task_name"],
    #     required_data_keys=required_data_keys,
    #     chunk_size=args["chunk_size"],
    #     norm_bound=FRANKA_JOINT_LIMITS,
    #     batch_size=batch_size,
    # )
    train_dataloader, val_dataloader = load_temporal_data(
        skill_or_task=task_name,
        data_dir=data_dir,
        chunk_size=100,
        norm_bound=FRANKA_JOINT_LIMITS,
        batch_size=batch_size,
    )

    # Save configuration
    config["rlbench_env"] = str(config["rlbench_env"])
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    json_path = os.path.join(ckpt_dir, f"config.json")
    with open(json_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def forward_pass(data, policy):
    images = data["images"]
    joint_action = data["joint_action"]
    is_pad = data["is_pad"]
    gripper_action = data["gripper_action"]
    task_emb = data["task_emb"]

    # Append gripper action to joint action
    action = torch.concatenate((joint_action, gripper_action.unsqueeze(-1)), 2)

    # Get the current joint state
    qpos = action[:, 0, :]

    images, qpos, action, is_pad, task_emb = (
        images.cuda(),
        qpos.cuda(),
        action.cuda(),
        is_pad.cuda(),
        task_emb.cuda(),
    )
    return policy(
        qpos, images, actions=action, is_pad=is_pad, task_emb=task_emb
    )  # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        if epoch % 250 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument(
        "--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True
    )
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument(
        "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "--batch_size", action="store", type=int, help="batch_size", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument(
        "--num_epochs", action="store", type=int, help="num_epochs", required=True
    )
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--hidden_dim", action="store", type=int, help="hidden_dim", required=False
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    # add this for multi-task embedding condition
    parser.add_argument("--multi_task", action="store_true")
    parser.add_argument("--ckpt_names", action="store", nargs="*", help="ckpt_names")
    parser.add_argument("--seq_skills", action="store_true")
    parser.add_argument("--data_dir", action="store")
    parser.add_argument("--transformer_only", action="store_true")
    parser.add_argument("--model_path_dict", type=json.loads)
    main(vars(parser.parse_args()))
