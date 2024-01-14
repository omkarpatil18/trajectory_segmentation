from typing import List
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, CustomDetectedCondition, GraspedCondition
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
import numpy as np

class SHockey(Task):

    def init_task(self) -> None:
        self.stick = Shape('hockey_stick')
        self.ball0 = Shape('hockey_ball0')
        self.ball1 = Shape('hockey_ball1')
        self.ball2 = Shape('hockey_ball2')
        self.success_detector = ProximitySensor('success')

        self.condition0 = GraspedCondition(self.robot.gripper, self.stick)
        self.condition1 = CustomDetectedCondition(self.ball0, self.success_detector)
        self.condition2 = CustomDetectedCondition(self.ball1, self.success_detector)

        self.register_success_conditions([
            self.condition0, self.condition1, self.condition2
            ])
        self.register_graspable_objects([self.stick])
        self.register_change_point_conditions([
            self.condition0, self.condition1, self.condition2
        ])

    def init_episode(self, index: int) -> List[str]:
        
        color_choice = np.random.choice(list(range(len(colors))),
            size=3, replace=False)
        color_names = []

        name, rgb = colors[color_choice[0]]
        self.ball0.set_color(rgb)
        color_names.append (name)

        name, rgb = colors[color_choice[1]]
        self.ball1.set_color(rgb)
        color_names.append (name)

        name, rgb = colors[color_choice[2]]
        self.ball2.set_color(rgb)
        
        self.register_instructions([
            [
                'Lift the hockey stick.',
                'Strike the %s ball into the net.' % color_names[0],
                'Aim to hit the %s ball into the goal.' % color_names[1]
            ],
            [
                'Retrieve the hockey stick.',
                'Swing at the %s ball to make it into the net.' % color_names[0],
                'Attempt to score by hitting the %s ball into the goal.' % color_names[1]
            ],
            [
                'Pick up the hockey stick.',
                'Skillfully hit the %s ball into the net.' % color_names[0],
                'Try to send the %s ball into the goal using the stick.' % color_names[1]
            ],
            [
                'Raise the hockey stick.',
                'Drive the %s ball into the net.' % color_names[0],
                'Strive to hit the %s ball into the goal.' % color_names[1]
            ],
            [
                'Take the hockey stick.',
                'Efficiently hit the %s ball into the net.' % color_names[0],
                'Your goal is to hit the %s ball into the goal.' % color_names[1]
            ]
        ])

        return ['hit the ball into the net',
                'use the stick to push the hockey ball into the goal',
                'pick up the hockey stick, then swing at the ball in the '
                'direction of the net',
                'score a hockey goal',
                'grasping one end of the hockey stick, swing it such that the '
                'other end collides with the ball such that the ball goes '
                'into the goal']

    def variation_count(self) -> int:
        return 1
