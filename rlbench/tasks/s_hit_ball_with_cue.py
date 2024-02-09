from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition, \
    ConditionSet, CustomDetectedCondition


class SHitBallWithCue(Task):

    def init_task(self) -> None:
        self.queue = Shape('queue')
        self.success_sensor = ProximitySensor('success')
        self.ball = Shape('ball')
        self.register_graspable_objects([self.queue])

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.queue),
            DetectedCondition(self.ball, self.success_sensor)
        ], order_matters=True)
        self.register_success_conditions([cond_set])

        self.register_instructions([
            [
                'Pick up the cue.',
                'Use the cue to guide the ball into the goal.'
            ],
            [
                'Retrieve the cue from its place.',
                'Employ the cue to direct the ball into the goal.'
            ],
            [
                'Gently lift the cue.',
                'Skillfully utilize the cue to place the ball into the goal.'
            ],
            [
                'Grasp the cue firmly.',
                'Employ the cue as a guide to insert the ball into the goal.'
            ],
            [
                'Lift the cue with care.',
                'Guide the ball into the goal using the cue.'
            ],
            [
                'SKILL_PICK',
                'SKILL_PUSH_BALL'
            ]
        ])


    def init_episode(self, index: int) -> List[str]:
        self.register_change_point_conditions([
            CustomDetectedCondition(self.queue, ProximitySensor('success1')),
            DetectedCondition(self.ball, self.success_sensor)
        ])
        return ['hit ball with queue in to the goal',
                'pot the ball in the goal',
                'pick up the que and use it to pot the ball into the goal']

    def variation_count(self) -> int:
        return 1
