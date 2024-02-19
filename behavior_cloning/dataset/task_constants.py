import sys
import re

sys.path.append("/home/local/ASUAD/opatil3/src/trajectory_segmentation/rlbench")
from tasks import OpenBox, SStackBlocks, SPutShoesInBox, SPutItemInDrawer
from constants import PICK_EMBEDDING_DICT, PLACE_EMBEDDING_DICT

### Task parameters
DATA_DIR = "/home/local/ASUAD/opatil3/datasets/rlbench"
IMG_SIZE = [224, 224]

SIM_TASK_CONFIG = {
    ########## stack blocks ##########
    # "sim_skill_pick": {
    "sim_skill_pick_red": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_red"],
        "skill_emb": None,
    },
    "sim_skill_pick_blue": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_blue"],
        "skill_emb": None,
    },
    "sim_skill_pick_green": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_green"],
        "skill_emb": None,
    },
    "sim_skill_pick_yellow": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_yellow"],
        "skill_emb": None,
    },
    ### Place skills
    "sim_skill_place_red": {
        "rlbench_env": None,
        "episode_len": 100,
        "train_subtasks": ["SKILL_PLACE_ON_red"],
        "skill_emb": None,
    },
    "sim_skill_place_blue": {
        "rlbench_env": None,
        "episode_len": 100,
        "train_subtasks": ["SKILL_PLACE_ON_blue"],
        "skill_emb": None,
    },
    "sim_skill_place_green": {
        "rlbench_env": None,
        "episode_len": 100,
        "train_subtasks": ["SKILL_PLACE_ON_green"],
        "skill_emb": None,
    },
    "sim_skill_place_green_center": {
        "rlbench_env": None,
        "episode_len": 100,
        "train_subtasks": ["SKILL_PLACE_ON_green_center"],
        "skill_emb": None,
    },
    ### Combined tasks
    "sim_stack_blocks": {
        "rlbench_env": SStackBlocks,
        "episode_len": 750,
        "train_subtasks": ["s_stack_blocks"],
        "lang_to_skill_map": {
            re.compile(
                r"(?=.*pick)(?=.*red)", flags=re.IGNORECASE
            ): "sim_skill_pick_red",
            re.compile(
                r"(?=.*pick)(?=.*blue)", flags=re.IGNORECASE
            ): "sim_skill_pick_blue",
            re.compile(
                r"(?=.*pick)(?=.*green)", flags=re.IGNORECASE
            ): "sim_skill_pick_green",
            re.compile(
                r"(?=.*pick)(?=.*yellow)", flags=re.IGNORECASE
            ): "sim_skill_pick_yellow",
            re.compile(
                r"(?=.*place)(?=.*red)", flags=re.IGNORECASE
            ): "sim_skill_place_red",
            re.compile(
                r"(?=.*place)(?=.*blue)", flags=re.IGNORECASE
            ): "sim_skill_place_blue",
            re.compile(
                r"(?=.*place)(?=.*green)", flags=re.IGNORECASE
            ): "sim_skill_place_green",
            re.compile(
                r"(?=.*place)(?=.*green)(?=.*center)", flags=re.IGNORECASE
            ): "sim_skill_place_green_center",
        },
        "skill_emb": None,
    },
    ########## shoe in box ##########
    "sim_skill_open_box": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_OPEN_box"],
        "skill_emb": None,
    },
    "sim_skill_pick_shoe1": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_shoe1"],
        "skill_emb": None,
    },
    "sim_skill_pick_shoe2": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_shoe2"],
        "skill_emb": None,
    },
    "sim_skill_place_shoe1": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PLACE_shoe1"],
        "skill_emb": None,
    },
    "sim_skill_place_shoe2": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PLACE_shoe2"],
        "skill_emb": None,
    },
    ### Combined tasks
    "sim_put_shoe_in_box": {
        "rlbench_env": SPutShoesInBox,
        "episode_len": 750,
        "train_subtasks": ["s_put_shoes_in_box"],
        "skill_emb": None,
    },
    ########## item in drawer ##########
    "sim_skill_open_drawer": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_OPEN_drawer"],
        "skill_emb": None,
    },
    "sim_skill_pick_item": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_item"],
        "skill_emb": None,
    },
    "sim_skill_place_item": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PLACE_item"],
        "skill_emb": None,
    },
    ### Combined tasks
    "sim_put_item_in_drawer": {
        "rlbench_env": SPutItemInDrawer,
        "episode_len": 750,
        "train_subtasks": ["s_put_item_in_drawer"],
        "skill_emb": None,
    },
}

model_path_dict = {
    "sim_skill_pick_red": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/pick_red_xxx",
    "sim_skill_pick_blue": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/pick_blue_xxx",
    "sim_skill_pick_yellow": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/pick_yellow_xxx",
    "sim_skill_pick_green": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/pick_green_xxx",
    "sim_skill_place_red": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/place_red_xxx",
    "sim_skill_place_blue": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/place_blue_xxx",
    "sim_skill_place_green": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/place_green_xxx",
    "sim_skill_place_green_center": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/place_green_center_xxx",
    "sim_skill_open_drawer": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_item_in_drawer/vxxx/open_drawer_xxx",
    "sim_skill_pick_item": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_item_in_drawer/vxxx/pick_item_xxx",
    "sim_skill_place_item": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_item_in_drawer/vxxx/place_item_xxx",
}
