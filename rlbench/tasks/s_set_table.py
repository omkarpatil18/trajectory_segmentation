from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, CustomConditionSet


class SSetTable(Task):

    def init_task(self) -> None:
        self.plate = Shape('plate')
        self.fork = Shape('fork')
        self.knife = Shape('knife')
        self.spoon = Shape('spoon')
        self.glass = Shape('glass')
        self.negate = ProximitySensor('negate')

        self.register_success_conditions([
            #DetectedCondition(plate, negate, negated = True),
            DetectedCondition(self.plate, ProximitySensor('plate_detector')),
            #DetectedCondition(fork, negate, negated = True),
            DetectedCondition(self.fork, ProximitySensor('fork_detector')),
            #DetectedCondition(knife, negate, negated = True),
            DetectedCondition(self.knife, ProximitySensor('knife_detector')),
            #DetectedCondition(spoon, negate, negated = True),
            DetectedCondition(self.spoon, ProximitySensor('spoon_detector')),
            #DetectedCondition(glass, negate, negated = True),
            DetectedCondition(self.glass, ProximitySensor('glass_detector')),
            NothingGrasped(self.robot.gripper)])
        self.register_graspable_objects([self.plate, self.fork, self.knife, self.spoon, self.glass])

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
            ],
            [
                'SKILL_PICK_plate',
                'SKILL_PLACE_plate',
                'SKILL_PICK_fork',
                'SKILL_PLACE_fork',
                'SKILL_PICK_knife',
                'SKILL_PLACE_knife',
                'SKILL_PICK_spoon',
                'SKILL_PLACE_spoon',
                'SKILL_PICK_glass',
                'SKILL_PLACE_glass'
            ]
        ])

    def init_episode(self, index: int) -> List[str]:
        self.register_change_point_conditions([
            DetectedCondition(self.plate, self.negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.plate, ProximitySensor('plate_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(self.fork, self.negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.fork, ProximitySensor('fork_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(self.knife, self.negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.knife, ProximitySensor('knife_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(self.spoon, self.negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.spoon, ProximitySensor('spoon_detector')),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(self.glass, self.negate, negated = True),
            CustomConditionSet([
                DetectedCondition(self.glass, ProximitySensor('glass_detector')),
                NothingGrasped(self.robot.gripper)
            ])            
        ])
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
