import sys
import re

sys.path.append("/home/local/ASUAD/opatil3/src/trajectory_segmentation/rlbench")
from tasks import OpenBox, SStackBlocks
from constants import PICK_EMBEDDING_DICT, PLACE_EMBEDDING_DICT

### Task parameters
DATA_DIR = "/home/local/ASUAD/opatil3/datasets/rlbench"
IMG_SIZE = [224, 224]

SIM_TASK_CONFIG = {
    "sim_open_box": {
        "train_subtasks": ["sim_open_box"],
        "rlbench_env": OpenBox,
        "episode_len": 250,
        "lang_to_skill_map": {},
        "skill_emb": None,
    },
    ### Pick skills
    "sim_skill_pick": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PICK_"],
        "skill_emb": PICK_EMBEDDING_DICT,
    },
    "sim_skill_pick_red": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PICK_red"],
        "skill_emb": None,
    },
    "sim_skill_pick_blue": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PICK_blue"],
        "skill_emb": None,
    },
    "sim_skill_pick_green": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PICK_green"],
        "skill_emb": None,
    },
    "sim_skill_pick_yellow": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PICK_yellow"],
        "skill_emb": None,
    },
    ### Place skills
    "sim_skill_place": {
        "rlbench_env": None,
        "train_subtasks": ["SKILL_PLACE_"],
        "episode_len": 250,
        "skill_emb": PLACE_EMBEDDING_DICT,
    },
    "sim_skill_place_red": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PLACE_red"],
        "skill_emb": None,
    },
    "sim_skill_place_blue": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PLACE_blue"],
        "skill_emb": None,
    },
    "sim_skill_place_green": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PLACE_green"],
        "skill_emb": None,
    },
    "sim_skill_place_yellow": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PLACE_yellow"],
        "skill_emb": None,
    },
    ### Combined tasks
    "sim_stack_blocks_mt": {
        "rlbench_env": SStackBlocks,
        "episode_len": 500,
        "train_subtasks": ["s_stack_blocks"],
        "skill_emb": None,
        "lang_to_skill_map": {
            re.compile(r"pick", flags=re.IGNORECASE): "sim_skill_pick",
            re.compile(r"place", flags=re.IGNORECASE): "sim_skill_place",
        },
    },
    "sim_stack_blocks": {
        "rlbench_env": SStackBlocks,
        "episode_len": 500,
        "train_subtasks": ["s_stack_blocks"],
        "lang_to_skill_map": {
            re.compile(
                r"(?=.*pick)(?=.*red)", flags=re.IGNORECASE
            ): "sim_skill_pick_red",
            # re.compile(
            #     r"(?=.*pick)(?=.*blue)", flags=re.IGNORECASE
            # ): "sim_skill_pick_blue",
            # re.compile(
            #     r"(?=.*pick)(?=.*green)", flags=re.IGNORECASE
            # ): "sim_skill_pick_green",
            # re.compile(
            #     r"(?=.*pick)(?=.*yellow)", flags=re.IGNORECASE
            # ): "sim_skill_pick_yellow",
            # re.compile(
            #     r"(?=.*place)(?=.*red)", flags=re.IGNORECASE
            # ): "sim_skill_place_red",
            # re.compile(
            #     r"(?=.*place)(?=.*blue)", flags=re.IGNORECASE
            # ): "sim_skill_place_blue",
            # re.compile(
            #     r"(?=.*place)(?=.*green)", flags=re.IGNORECASE
            # ): "sim_skill_place_green",
            # re.compile(
            #     r"(?=.*place)(?=.*yellow)", flags=re.IGNORECASE
            # ): "sim_skill_place_yellow",
        },
        "skill_emb": None,
    },
}
