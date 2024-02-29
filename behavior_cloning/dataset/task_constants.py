import sys
import re

sys.path.append("/home/local/ASUAD/opatil3/src/trajectory_segmentation/rlbench")
from tasks import (
    OpenBox,
    SStackBlocks,
    SPutShoesInBox,
    SPutItemInDrawer,
    SBallInHoop,
    SStackCups,
)
from constants import PICK_EMBEDDING_DICT, PLACE_EMBEDDING_DICT

### Task parameters
DATA_DIR = "/home/local/ASUAD/opatil3/datasets/rlbench"
IMG_SIZE = [224, 224]

SIM_TASK_CONFIG = {
    ########## stack blocks ##########
    # "sim_skill_pick": {
    "sim_skill_pick_red_block": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_red"],
        "skill_emb": None,
    },
    "sim_skill_pick_green_block": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_green"],
        "skill_emb": None,
    },
    ### Place skills
    "sim_skill_place_red_block": {
        "rlbench_env": None,
        "episode_len": 100,
        "train_subtasks": ["SKILL_PLACE_ON_red"],
        "skill_emb": None,
    },
    "sim_skill_place_green_center": {
        "rlbench_env": None,
        "episode_len": 100,
        "train_subtasks": ["SKILL_PLACE_ON_green"],
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
            ): "sim_skill_pick_red_block",
            re.compile(
                r"(?=.*pick)(?=.*green)", flags=re.IGNORECASE
            ): "sim_skill_pick_green_block",
            re.compile(
                r"(?=.*place)(?=.*red)", flags=re.IGNORECASE
            ): "sim_skill_place_red_block",
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
    "sim_put_shoes_in_box": {
        "rlbench_env": SPutShoesInBox,
        "episode_len": 750,
        "train_subtasks": ["s_put_shoes_in_box"],
        "skill_emb": None,
        "lang_to_skill_map": {
            re.compile(
                r"(?=.*pick)(?=.*right shoe)", flags=re.IGNORECASE
            ): "sim_skill_pick_shoe1",
            re.compile(
                r"(?=.*pick)(?=.*left shoe)", flags=re.IGNORECASE
            ): "sim_skill_pick_shoe2",
            re.compile(
                r"(?=.*place)(?=.*right shoe)", flags=re.IGNORECASE
            ): "sim_skill_place_shoe1",
            re.compile(
                r"(?=.*place)(?=.*left shoe)", flags=re.IGNORECASE
            ): "sim_skill_place_shoe2",
            re.compile(
                r"(?=.*open)(?=.*box)", flags=re.IGNORECASE
            ): "sim_skill_open_box",
        },
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
        "lang_to_skill_map": {
            re.compile(
                r"(?=.*pick)(?=.*block)", flags=re.IGNORECASE
            ): "sim_skill_pick_item",
            re.compile(
                r"(?=.*place)(?=.*block)", flags=re.IGNORECASE
            ): "sim_skill_place_item",
            re.compile(
                r"(?=.*open)(?=.*drawer)", flags=re.IGNORECASE
            ): "sim_skill_open_drawer",
        },
    },
    ########## ball in hoop ##########
    "sim_skill_pick_ball": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_ball"],
        "skill_emb": None,
    },
    "sim_skill_place_ball": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PLACE_ball"],
        "skill_emb": None,
    },
    ### Combined tasks
    "sim_ball_in_hoop": {
        "rlbench_env": SBallInHoop,
        "episode_len": 500,
        "train_subtasks": ["s_ball_in_hoop"],
        "skill_emb": None,
        "lang_to_skill_map": {
            re.compile(
                r"(?=.*pick)(?=.*basketball)", flags=re.IGNORECASE
            ): "sim_skill_pick_ball",
            re.compile(
                r"(?=.*place)(?=.*basketball)", flags=re.IGNORECASE
            ): "sim_skill_place_ball",
        },
    },
    ########## stack cups ##########
    "sim_skill_pick_green_cup": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_green"],
        "skill_emb": None,
    },
    "sim_skill_pick_blue_cup": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PICK_blue"],
        "skill_emb": None,
    },
    "sim_skill_place_red_cup": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PLACE_red"],
        "skill_emb": None,
    },
    "sim_skill_place_green_cup": {
        "rlbench_env": None,
        "episode_len": 200,
        "train_subtasks": ["SKILL_PLACE_green"],
        "skill_emb": None,
    },
    ### Combined tasks
    "sim_stack_cups": {
        "rlbench_env": SStackCups,
        "episode_len": 750,
        "train_subtasks": ["s_stack_cups"],
        "skill_emb": None,
        "lang_to_skill_map": {
            re.compile(
                r"(?=.*pick)(?=.*green)", flags=re.IGNORECASE
            ): "sim_skill_pick_green_cup",
            re.compile(
                r"(?=.*pick)(?=.*blue)", flags=re.IGNORECASE
            ): "sim_skill_pick_blue_cup",
            re.compile(
                r"(?=.*place)(?=.*green cup on)", flags=re.IGNORECASE
            ): "sim_skill_place_red_cup",
            re.compile(
                r"(?=.*place)(?=.*blue cup on)", flags=re.IGNORECASE
            ): "sim_skill_place_green_cup",
        },
    },
}

model_path_dict = {
    # Stack blocks
    "sim_skill_pick_red_block": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/pick_red_block_xxx",
    "sim_skill_pick_green_block": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/pick_green_block_xxx",
    "sim_skill_place_red_block": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/place_red_block_xxx",
    "sim_skill_place_green_center": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_blocks/vxxx/place_green_center_xxx",
    # Stack cups
    "sim_skill_pick_green_cup": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_cups/vxxx/pick_green_cup2_xxx",
    "sim_skill_pick_blue_cup": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_cups/vxxx/pick_blue_cup2_xxx",
    "sim_skill_place_red_cup": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_cups/vxxx/place_red_cup2_xxx",
    "sim_skill_place_green_cup": "/home/local/ASUAD/opatil3/checkpoints/mt_act/stack_cups/vxxx/place_green_cup2_xxx",
    # Put item in drawer
    "sim_skill_open_drawer": "/home/local/ASUAD/opatil3/checkpoints/mt_act/seg_put_item_in_drawer/vxxx/open_drawer_xxx",
    "sim_skill_pick_item": "/home/local/ASUAD/opatil3/checkpoints/mt_act/seg_put_item_in_drawer/vxxx/pick_item_xxx",
    "sim_skill_place_item": "/home/local/ASUAD/opatil3/checkpoints/mt_act/seg_put_item_in_drawer/vxxx/place_item_xxx",
    # Ball in hoop
    "sim_skill_pick_ball": "/home/local/ASUAD/opatil3/checkpoints/mt_act/ball_in_hoop/vxxx/pick_ball_xxx",
    "sim_skill_place_ball": "/home/local/ASUAD/opatil3/checkpoints/mt_act/ball_in_hoop/vxxx/place_ball_xxx",
    # Shoe in box
    "sim_skill_pick_shoe1": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_shoes_in_box/vxxx/pick_shoe1_xxx",
    "sim_skill_place_shoe1": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_shoes_in_box/vxxx/place_shoe1_xxx",
    "sim_skill_pick_shoe2": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_shoes_in_box/vxxx/pick_shoe2_xxx",
    "sim_skill_place_shoe2": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_shoes_in_box/vxxx/place_shoe2_xxx",
    "sim_skill_open_box": "/home/local/ASUAD/opatil3/checkpoints/mt_act/put_shoes_in_box/vxxx/open_box_xxx",
}
