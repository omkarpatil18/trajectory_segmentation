import sys

sys.path.append("/home/local/ASUAD/opatil3/src/trajectory_segmentation/rlbench")
from tasks import PutItemInDrawer, OpenBox, OpenDrawer, SStackBlocks
from constants import PICK_EMBEDDING_DICT, PLACE_EMBEDDING_DICT

### Task parameters
DATA_DIR = "/home/local/ASUAD/opatil3/datasets/rlbench"
# DATA_DIR = "/home/local/ASUAD/opatil3/datasets/temporal/data/s_stack_blocks/variation0/episodes/"

SIM_TASK_CONFIG = {
    "sim_open_box": {
        "train_subtasks": ["sim_open_box"],
        "rlbench_env": OpenBox,
        "episode_len": 250,
        "skill_sequence": [],
        "skill_emb": None,
    },
    "sim_skill_pick": {
        "rlbench_env": None,
        "episode_len": 250,
        "train_subtasks": ["SKILL_PICK_"],
        "skill_emb": PICK_EMBEDDING_DICT,
    },
    "sim_skill_place": {
        "rlbench_env": None,
        "train_subtasks": ["SKILL_PLACE_"],
        "episode_len": 250,
        "skill_emb": PLACE_EMBEDDING_DICT,
    },
    "sim_stack_blocks": {
        "rlbench_env": SStackBlocks,
        "episode_len": 250,
        "train_subtasks": ["s_stack_blocks"],
        "skill_sequence": [
            "sim_skill_pick",
            "sim_skill_place",
        ],
        "skill_emb": None,
    },
}
