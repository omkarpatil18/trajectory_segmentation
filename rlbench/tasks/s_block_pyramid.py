from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import (DetectedSeveralCondition,
                                        NothingGrasped, ConditionSet)
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors


class SBlockPyramid(Task):

    def init_task(self) -> None:
        self.blocks = [Shape('block_pyramid_block%d' % i) for i in range(6)]
        self.distractors = [Shape(
            'block_pyramid_distractor_block%d' % i) for i in range(6)]
        success_detectors = [ProximitySensor(
            'block_pyramid_success_block%d' % i) for i in range(3)]

        cond_set = ConditionSet([
            DetectedSeveralCondition(self.blocks, success_detectors[0], 3),
            DetectedSeveralCondition(self.blocks, success_detectors[1], 2),
            DetectedSeveralCondition(self.blocks, success_detectors[2], 1),
            NothingGrasped(self.robot.gripper)
        ])
        self.register_success_conditions([cond_set])
        
        self.register_change_point_conditions([
            DetectedSeveralCondition(self.blocks, success_detectors[0], 3),
            DetectedSeveralCondition(self.blocks, success_detectors[1], 2),
            DetectedSeveralCondition(self.blocks, success_detectors[2], 1)
        ])

        self.register_graspable_objects(self.blocks + self.distractors)
        self.spawn_boundary = SpawnBoundary(
            [Shape('block_pyramid_boundary%d' % i) for i in range(4)])

    def init_episode(self, index: int) -> List[str]:

        color_name_1, color_rgb_1 = colors[index]
        for obj in self.blocks[:3]:
            obj.set_color(color_rgb_1)

        color_choice = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)

        color_name_2, color_rgb_2 = colors[ color_choice[0] ]
        for obj in self.blocks[3:5]:
            obj.set_color(color_rgb_2)
    
        color_name_3, color_rgb_3 = colors[ color_choice[1] ]
        for obj in self.blocks[5:]:
            obj.set_color(color_rgb_3)
    
        color_choice = np.random.choice(
            list(range(len(colors))),
            size=6, replace=False)

        for color_choice, obj in enumerate(self.distractors):
            name, rgb = colors[color_choice]
            obj.set_color(rgb)

        self.spawn_boundary.clear()
        for ob in self.blocks + self.distractors:
            self.spawn_boundary.sample(
                ob, min_distance=0.08, min_rotation=(0.0, 0.0, -np.pi / 4),
                max_rotation=(0.0, 0.0, np.pi / 4))

        self.register_instructions([
            ['Create base of the pyramid with 3 %s blocks' % color_name_1, 
            'Create the next layer of pyramid with 2 %s blocks' % color_name_2,
            'Create the top of pyramid with a %s block' % color_name_3],

            ['Build the pyramid base using 3 %s blocks.' % color_name_1, 
            'Construct the next layer with 2 %s blocks.' % color_name_2, 
            'Cap the pyramid with a single %s block.' % color_name_3],

            ['Place 3 %s blocks to form the pyramid base.' % color_name_1, 
            'Assemble the second layer using 2 %s blocks.' % color_name_2, 
            'Complete the pyramid top with a single %s block.' % color_name_3],

            ['Start the pyramid with 3 %s blocks for the base.' % color_name_1, 
            'Continue with the second layer using 2 %s blocks.' % color_name_2, 
            'Finish the pyramid top using a %s block.' % color_name_3],

            ['Use 3 %s blocks to create the pyramid base.' % color_name_1, 
            'Add the second layer with 2 %s blocks.' % color_name_2, 
            'Top off the pyramid with a single %s block.' % color_name_3],

            ['Begin the pyramid base with 3 %s blocks.' % color_name_1, 
            'Extend to the next layer with 2 %s blocks.' % color_name_2, 
            'Complete the pyramid top with a %s block.' % color_name_3]
        ])
        
        return ['stack %s blocks in a pyramid' % color_name_1,
                'create a pyramid with the %s objects' % color_name_1,
                'make a pyramid out of %s cubes' % color_name_1,
                'position the %s blocks in the shape of a pyramid' % color_name_1,
                'use the %s blocks to build a pyramid' % color_name_1]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]