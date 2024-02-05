from rlbench.tasks import PutItemInDrawer, OpenBox, OpenDrawer

### Task parameters
DATA_DIR = "/home/local/ASUAD/opatil3/datasets/rlbench"
SIM_TASK_CONFIG = {
    "sim_put_item_in_drawer": {
        "rlbench_env": PutItemInDrawer,
        "episode_len": 250,
        "skill_sequence": [
            "sim_skill_open_drawer",
            "sim_skill_pick",
            "sim_skill_place",
        ],
    },
    "sim_open_box": {
        "train_subtasks": ["sim_open_box"],
        "rlbench_env": OpenBox,
        "episode_len": 250,
        "skill_sequence": [],
    },
}
