from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, CustomConditionSet


class SSetTable(Task):

    def init_task(self) -> None:
        plate = Shape('plate')
        fork = Shape('fork')
        knife = Shape('knife')
        spoon = Shape('spoon')
        glass = Shape('glass')
        negate = ProximitySensor('negate')

        self.register_success_conditions([
            #DetectedCondition(plate, negate, negated = True),
            DetectedCondition(plate, ProximitySensor('plate_detector')),
            #DetectedCondition(fork, negate, negated = True),
            DetectedCondition(fork, ProximitySensor('fork_detector')),
            #DetectedCondition(knife, negate, negated = True),
            DetectedCondition(knife, ProximitySensor('knife_detector')),
            #DetectedCondition(spoon, negate, negated = True),
            DetectedCondition(spoon, ProximitySensor('spoon_detector')),
            #DetectedCondition(glass, negate, negated = True),
            DetectedCondition(glass, ProximitySensor('glass_detector')),
            NothingGrasped(self.robot.gripper)])
        self.register_graspable_objects([plate, fork, knife, spoon, glass])

        self.register_change_point_conditions([
            DetectedCondition(plate, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(plate, ProximitySensor('plate_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(fork, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(fork, ProximitySensor('fork_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(knife, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(knife, ProximitySensor('knife_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(spoon, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(spoon, ProximitySensor('spoon_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(glass, negate, negated = True),
            CustomConditionSet([
                DetectedCondition(glass, ProximitySensor('glass_detector')),
                NothingGrasped(self.robot.gripper)
            ])            
        ])

        self.register_instructions([
            [
                'Retrieve the plate from the utensils holder.',
                'Position the plate on the table.',
                'Take the fork from the utensils holder.',
                'Place the fork on the left side of the plate.',
                'Pick up the knife from the utensils holder.',
                'Position the knife on the right side of the plate.',
                'Collect the spoon from the utensils holder.',
                'Set the spoon on the right side of the plate.',
                'Pick up the glass.',
                'Put the glass on the top right side of the plate.'
            ],
            [
                'Pick up the plate from the utensils holder.',
                'Place the plate on the table.',
                'Get the fork from the utensils holder.',
                'Position the fork on the left side of the plate.',
                'Take the knife from the utensils holder.',
                'Set the knife on the right side of the plate.',
                'Retrieve the spoon from the utensils holder.',
                'Put the spoon on the right side of the plate.',
                'Take the glass.',
                'Position the glass on the top right side of the plate.'
            ],
            [
                'Get the plate from the utensils holder.',
                'Put the plate on the table.',
                'Retrieve the fork from the utensils holder.',
                'Place the fork on the left side of the plate.',
                'Pick up the knife from the utensils holder.',
                'Set the knife on the right side of the plate.',
                'Take the spoon from the utensils holder.',
                'Position the spoon on the right side of the plate.',
                'Pick up the glass.',
                'Position the glass on the top right side of the plate.'
            ],
            [
                'Take the plate from the utensils holder.',
                'Position the plate on the table.',
                'Collect the fork from the utensils holder.',
                'Put the fork on the left side of the plate.',
                'Retrieve the knife from the utensils holder.',
                'Place the knife on the right side of the plate.',
                'Pick up the spoon from the utensils holder.',
                'Put the spoon on the right side of the plate.',
                'Take the glass.',
                'Place the glass on the top right side of the plate.'
            ],
            [
                'Retrieve the plate from the utensils holder.',
                'Position the plate on the table.',
                'Get the fork from the utensils holder.',
                'Put the fork on the left side of the plate.',
                'Take the knife from the utensils holder.',
                'Set the knife on the right side of the plate.',
                'Pick up the spoon from the utensils holder.',
                'Place the spoon on the right side of the plate.',
                'Pick up the glass.',
                'Place the glass on the top right side of the plate.'
            ]
        ])

    def init_episode(self, index: int) -> List[str]:
        return ['set the table'
                'place the dishes and cutlery on the table in preparation for '
                'a meal',
                'pick up the plate and put it down on the table, then place '
                'the fork to its left, the knife and then the spoon to its '
                'right, and set the glass down just above them',
                'prepare the table for a meal',
                'arrange the plate, cutlery and glass neatly on the table '
                'so that a person can eat',
                'get the table ready for lunch',
                'get the table ready for dinner']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
