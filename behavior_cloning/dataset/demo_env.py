import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, JointPosition
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseDrawer, PutItemInDrawer
import time

obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(
        arm_action_mode=JointPosition(), gripper_action_mode=Discrete()
    ),
    obs_config=ObservationConfig(),
    robot_setup="panda",
    headless=False,
)
env.launch()

steps_per_task = 100
num_episodes = 1

task = env.get_task(PutItemInDrawer)
task.sample_variation()  # random variation
demo = task.get_demos(1, live_demos=True)

for e in range(num_episodes):
    print("Reset Episode")
    descriptions, obs = task.reset()
    # breakpoint()
    for i in range(steps_per_task):
        action = np.random.random(env.action_shape)
        start = time.perf_counter()
        obs, reward, terminate = task.step(action)
        print(f"step time: {time.perf_counter()-start}")

print("Done")
env.shutdown()
