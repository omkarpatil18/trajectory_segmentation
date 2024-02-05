from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import (DetectedCondition, NothingGrasped,
                            CustomConditionSet, CustomDetectedCondition)
from rlbench.backend.spawn_boundary import SpawnBoundary


class SStackCups(Task):

    def init_task(self) -> None:
        success_sensor = ProximitySensor('success')
        negate = ProximitySensor('negate')

        self.cup1 = Shape('cup1')
        self.cup2 = Shape('cup2')
        self.cup3 = Shape('cup3')
        self.cup1_visual = Shape('cup1_visual')
        self.cup2_visual = Shape('cup2_visual')
        self.cup3_visaul = Shape('cup3_visual')

        self.boundary = SpawnBoundary([Shape('boundary')])

        self.register_graspable_objects([self.cup1, self.cup2, self.cup3])
        self.register_success_conditions([
            DetectedCondition(self.cup1, success_sensor),
            DetectedCondition(self.cup3, success_sensor),
            NothingGrasped(self.robot.gripper)
        ])
        self.register_change_point_conditions([
            CustomDetectedCondition(self.cup1, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.cup1, success_sensor),
                NothingGrasped(self.robot.gripper)
            ]),
            CustomDetectedCondition(self.cup3, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.cup3, success_sensor),
                NothingGrasped(self.robot.gripper)
            ])
        ])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        index = np.random.choice(len(colors))
        target_color_name, target_rgb = colors[index]

        random_idx = np.random.choice(len(colors))
        while random_idx == index:
            random_idx = np.random.choice(len(colors))
        other1_name, other1_rgb = colors[random_idx]

        random_idx = np.random.choice(len(colors))
        while random_idx == index:
            random_idx = np.random.choice(len(colors))
        other2_name, other2_rgb = colors[random_idx]

        self.cup2_visual.set_color(target_rgb)
        self.cup1_visual.set_color(other1_rgb)
        self.cup3_visaul.set_color(other2_rgb)

        self.boundary.clear()
        self.boundary.sample(self.cup2, min_distance=0.05,
                             min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        self.boundary.sample(self.cup1, min_distance=0.05,
                             min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        self.boundary.sample(self.cup3, min_distance=0.05,
                             min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        
        self.register_instructions([
            [
                'Pick up the %s cup.' % other1_name,
                'Position the %s cup on top of the %s cup.' % (other1_name, target_color_name),
                'Grab the %s cup.' % other2_name,
                'Place the %s cup on the %s cup.' % (other2_name, other1_name)
            ],
            [
                'Retrieve the %s cup.' % other1_name,
                'Set the %s cup on top of the %s cup.' % (other1_name, target_color_name),
                'Take the %s cup.' % other2_name,
                'Position the %s cup on the %s cup.' % (other2_name, other1_name)
            ],
            [
                'Pick up the %s cup.' % other1_name,
                'Place the %s cup on top of the %s cup.' % (other1_name, target_color_name),
                'Retrieve the %s cup.' % other2_name,
                'Set the %s cup on the %s cup.' % (other2_name, other1_name)
            ],
            [
                'Grab the %s cup.' % other1_name,
                'Position the %s cup on top of the %s cup.' % (other1_name, target_color_name),
                'Take the %s cup.' % other2_name,
                'Place the %s cup on the %s cup.' % (other2_name, other1_name)
            ],
            [
                'Take the %s cup.' % other1_name,
                'Set the %s cup on top of the %s cup.' % (other1_name, target_color_name),
                'Retrieve the %s cup.' % other2_name,
                'Position the %s cup on the %s cup.' % (other2_name, other1_name)
            ]
        ])

        return ['stack the other cups on top of the %s cup' % target_color_name,
                'place two of the cups onto the odd cup out',
                'put the remaining two cups on top of the %s cup'
                % target_color_name,
                'pick up and set the cups down into the %s cup'
                % target_color_name,
                'create a stack of cups with the %s cup as its base'
                % target_color_name,
                'keeping the %s cup on the table, stack the other two onto it'
                % target_color_name]

    def variation_count(self) -> int:
        return 1
