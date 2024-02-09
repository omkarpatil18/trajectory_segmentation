from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import (DetectedSeveralCondition,
                                        NothingGrasped, ConditionSet, 
                                        DetectedCondition, CustomConditionSet, CustomDetectedCondition)
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors


class SBlockPyramid(Task):

    def init_task(self) -> None:
        self.blocks = [Shape('block_pyramid_block%d' % i) for i in range(6)]
        self.distractors = [Shape(
            'block_pyramid_distractor_block%d' % i) for i in range(6)]
        self.success_detectors = [ProximitySensor(
            'block_pyramid_success_block%d' % i) for i in range(3)]
        self.negate = ProximitySensor('negate0')

        self.register_graspable_objects(self.blocks + self.distractors)
        self.spawn_boundary = SpawnBoundary(
            [Shape('block_pyramid_boundary%d' % i) for i in range(4)])

    def init_episode(self, index: int) -> List[str]:

        color_choice = np.random.choice(
            list(range(len(colors))), size=6, replace=False)

        color_names = []
        for i, obj in enumerate (self.blocks):
            name, rgb = colors[ color_choice[i] ]
            obj.set_color (rgb)
            color_names.append (name)
        
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
        
        cond_negate = [
            CustomDetectedCondition(self.blocks[i], self.negate, negated = True)
            for i in range (6)
        ]
        
        cond_set = ConditionSet(cond_negate + [
            DetectedSeveralCondition(self.blocks, self.success_detectors[0], 3),
            DetectedSeveralCondition(self.blocks, self.success_detectors[1], 2),
            DetectedSeveralCondition(self.blocks, self.success_detectors[1], 2),
            DetectedSeveralCondition(self.blocks, self.success_detectors[2], 1),
            NothingGrasped(self.robot.gripper)
        ])
        self.register_success_conditions([cond_set])
        
        self.register_change_point_conditions([
            cond_negate[0],
            CustomConditionSet([
                DetectedSeveralCondition(self.blocks, self.success_detectors[0], 1),
                NothingGrasped(self.robot.gripper)
            ]),

            cond_negate[1],
            CustomConditionSet([
                DetectedSeveralCondition(self.blocks, self.success_detectors[0], 2),
                NothingGrasped(self.robot.gripper)
            ]),

            cond_negate[2],
            CustomConditionSet([
                DetectedSeveralCondition(self.blocks, self.success_detectors[0], 3),
                NothingGrasped(self.robot.gripper)
            ]),
            
            cond_negate[3],
            CustomConditionSet([
                DetectedSeveralCondition(self.blocks, self.success_detectors[1], 1),
                NothingGrasped(self.robot.gripper)
            ]),
            
            cond_negate[4],
            CustomConditionSet([
                DetectedSeveralCondition(self.blocks, self.success_detectors[1], 2),
                NothingGrasped(self.robot.gripper)
            ]),
            
            cond_negate[5],
            CustomConditionSet([
                DetectedSeveralCondition(self.blocks, self.success_detectors[2], 1),
                NothingGrasped(self.robot.gripper)
            ])
        ])

        self.register_instructions([
            [
                'Pick up a %s block.' % color_names[0],
                'Place the %s block in the center of the green strip.' % color_names[0],
                'Pick up a %s block.' % color_names[1],
                'Position the %s block to the right side of the green strip.' % color_names[1],
                'Take a %s block.' % color_names[2],
                'Set the %s block to the left side of the green strip.' % color_names[2],
                'Grab a %s block.' % color_names[3],
                'Place the %s block as the second layer on the right side.' % color_names[3],
                'Retrieve a %s block.' % color_names[4],
                'Position the %s block as the second layer on the left side.' % color_names[4],
                'Pick up a %s block.' % color_names[5],
                'Place the %s block as the third layer on the top of the pyramid.' % color_names[5]
            ],
            [
                'Take a %s block.' % color_names[0],
                'Position the %s block in the center of the green strip.' % color_names[0],
                'Retrieve a %s block.' % color_names[1],
                'Place the %s block to the right side of the green strip.' % color_names[1],
                'Grab a %s block.' % color_names[2],
                'Set the %s block to the left side of the green strip.' % color_names[2],
                'Pick up a %s block.' % color_names[3],
                'Position the %s block as the second layer on the right side.' % color_names[3],
                'Take a %s block.' % color_names[4],
                'Place the %s block as the second layer on the left side.' % color_names[4],
                'Retrieve a %s block.' % color_names[5],
                'Position the %s block as the third layer on the top of the pyramid.' % color_names[5]
            ],
            [
                'Grab a %s block.' % color_names[0],
                'Place the %s block in the center of the green strip.' % color_names[0],
                'Take a %s block.' % color_names[1],
                'Position the %s block to the right side of the green strip.' % color_names[1],
                'Retrieve a %s block.' % color_names[2],
                'Set the %s block to the left side of the green strip.' % color_names[2],
                'Pick up a %s block.' % color_names[3],
                'Place the %s block as the second layer on the right side.' % color_names[3],
                'Grab a %s block.' % color_names[4],
                'Position the %s block as the second layer on the left side.' % color_names[4],
                'Take a %s block.' % color_names[5],
                'Place the %s block as the third layer on the top of the pyramid.' % color_names[5]
            ],
            [
                'Retrieve a %s block.' % color_names[0],
                'Position the %s block in the center of the green strip.' % color_names[0],
                'Pick up a %s block.' % color_names[1],
                'Place the %s block to the right side of the green strip.' % color_names[1],
                'Grab a %s block.' % color_names[2],
                'Set the %s block to the left side of the green strip.' % color_names[2],
                'Take a %s block.' % color_names[3],
                'Position the %s block as the second layer on the right side.' % color_names[3],
                'Retrieve a %s block.' % color_names[4],
                'Place the %s block as the second layer on the left side.' % color_names[4],
                'Pick up a %s block.' % color_names[5],
                'Place the %s block as the third layer on the top of the pyramid.' % color_names[5]
            ],
            [
                'Take a %s block.' % color_names[0],
                'Position the %s block in the center of the green strip.' % color_names[0],
                'Retrieve a %s block.' % color_names[1],
                'Place the %s block to the right side of the green strip.' % color_names[1],
                'Grab a %s block.' % color_names[2],
                'Set the %s block to the left side of the green strip.' % color_names[2],
                'Pick up a %s block.' % color_names[3],
                'Position the %s block as the second layer on the right side.' % color_names[3],
                'Take a %s block.' % color_names[4],
                'Place the %s block as the second layer on the left side.' % color_names[4],
                'Retrieve a %s block.' % color_names[5],
                'Place the %s block as the third layer on the top of the pyramid.' % color_names[5]
            ],
            [
                'SKILL_PICK_%s' % color_names[0],
                'SKILL_PLACE_BOTTOM_CENTER',
                'SKILL_PICK_%s' % color_names[1],
                'SKILL_PLACE_BOTTOM_RIGHT',
                'SKILL_PICK_%s' % color_names[2],
                'SKILL_PLACE_BOTTOM_LEFT',
                'SKILL_PICK_%s' % color_names[3],
                'SKILL_PLACE_MIDDLE_RIGHT',
                'SKILL_PICK_%s' % color_names[4],
                'SKILL_PLACE_MIDDLE_LEFT',
                'SKILL_PICK_%s' % color_names[5],
                'SKILL_PLACE_TOP',
            ]
        ])

        color_name_1 = color_names[0]
        return ['stack %s blocks in a pyramid' % color_name_1,
                'create a pyramid with the %s objects' % color_name_1,
                'make a pyramid out of %s cubes' % color_name_1,
                'position the %s blocks in the shape of a pyramid' % color_name_1,
                'use the %s blocks to build a pyramid' % color_name_1]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]