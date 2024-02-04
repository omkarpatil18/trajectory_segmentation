from typing import List

import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors


class SSlideObjects(Task):

    def init_task(self) -> None:
        self._block = [Shape('block0'), Shape('block1'), Shape('block2')]
        self._target = [
            ProximitySensor('success0'),
            ProximitySensor('success1'),
            ProximitySensor('success2')]

        self._target_obj = [
            Shape('target0'), Shape('target1'), Shape('target2')
        ]

        self.register_success_conditions([
            DetectedCondition(self._block[0], self._target[0]),
            DetectedCondition(self._block[1], self._target[1]),
            DetectedCondition(self._block[2], self._target[2])])
        
        self.register_change_point_conditions([
            DetectedCondition(self._block[0], self._target[0]),
            DetectedCondition(self._block[1], self._target[1]),
            DetectedCondition(self._block[2], self._target[2])
        ])

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        
        color_choice = np.random.choice(list(range(len(colors))),
            size=6, replace=False)
        color_names = []

        for i in range(3):
            name, rgb = colors[color_choice[i]]
            color_names.append (name)
            self._block[i].set_color (rgb)
            self._target_obj[i].set_color(rgb)

        #for i in range (3):
        #    name, rgb = colors[color_choice[i + 3]]
        #    color_names.append (name)
        #    self._target_obj[i].set_color(rgb)
        
        self.register_instructions([
            [
                'Move the %s block towards the %s target.' % (color_names[0], color_names[0]),
                'Gently push the %s block to the %s target.' % (color_names[1], color_names[1]),
                'Slide the %s block to reach the %s target.' % (color_names[2], color_names[2])
            ],
            [
                'Push the %s block towards the %s target.' % (color_names[0], color_names[0]),
                'Guide the %s block to the %s target.' % (color_names[1], color_names[1]),
                'Navigate the %s block to the %s target.' % (color_names[2], color_names[2])
            ],
            [
                'Slide the %s block towards the %s target.' % (color_names[0], color_names[0]),
                'Apply pressure to push the %s block to the %s target.' % (color_names[1], color_names[1]),
                'Move the %s block steadily towards the %s target.' % (color_names[2], color_names[2])
            ],
            [
                'Gently push the %s block to the %s target.' % (color_names[0], color_names[0]),
                'Guide the %s block towards the %s target.' % (color_names[1], color_names[1]),
                'Push the %s block to the %s target carefully.' % (color_names[2], color_names[2])
            ],
            [
                'Push the %s block towards the %s target.' % (color_names[0], color_names[0]),
                'Direct the %s block to the %s target.' % (color_names[1], color_names[1]),
                'Slide the %s block to the %s target smoothly.' % (color_names[2], color_names[2])
            ]
        ])

        return ['slide the block to target',
                'slide the block onto the target',
                'push the block until it is sitting on top of the target',
                'slide the block towards the green target',
                'cover the target with the block by pushing the block in its'
                ' direction']

    def variation_count(self) -> int:
        return 1

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        return np.concatenate([
            self._block[0].get_position(), self._target[0].get_position()])

    def reward(self) -> float:
        grip_to_block = -np.linalg.norm(
            self._block[0].get_position() - self.robot.arm.get_tip().get_position())
        block_to_target = -np.linalg.norm(
            self._block[0].get_position() - self._target[0].get_position())
        return grip_to_block + block_to_target
