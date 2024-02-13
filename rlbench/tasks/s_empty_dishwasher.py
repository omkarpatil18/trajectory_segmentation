from typing import List, Tuple
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, CustomConditionSet


class SEmptyDishwasher(Task):

    def init_task(self) -> None:
        self.success_detector = ProximitySensor('success')
        self.success_plate = ProximitySensor('success_plate')
        self.success_door = ProximitySensor('success_door')

        self.door = Shape('dishwasher_door')
        self.plate = Shape('dishwasher_plate')
        self.tray = Shape('dishwasher_tray')

        self.register_graspable_objects([self.plate])
        
        self.register_success_conditions(
            [DetectedCondition(self.door, self.success_door),
            DetectedCondition(self.tray, self.success_door),
            DetectedCondition(self.plate, self.success_detector, negated=True),
            DetectedCondition(self.plate, self.success_plate)])

        self.register_instructions([
            [
                'Open the dishwasher door.',
                'Pull out the dishwasher tray.',
                'Lift the plate.',
                'Place the plate on top of the dishwasher.'
            ],
            [
                'Begin by opening the dishwasher door.',
                'Extend the dishwasher tray outward.',
                'Take the plate in your hands.',
                'Position the plate on the top of the dishwasher.'
            ],
            [
                'First, open the door of the dishwasher.',
                'Gently pull out the tray from the dishwasher.',
                'Grasp the plate.',
                'Carefully place the plate on the dishwasher top.'
            ],
            [
                'Initiate by opening the dishwasher door.',
                'Extend the dishwasher tray outwards.',
                'Pick up the plate with your hands.',
                'Put the plate on the top surface of the dishwasher.'
            ],
            [
                'Commence by opening the dishwasher door.',
                'Pull out the tray from the dishwasher.',
                'Take hold of the plate.',
                'Position the plate on the top of the dishwasher.'
            ],
            [
                'SKILL_OPEN',
                'SKILL_SLIDE',
                'SKILL_PICK',
                'SKILL_PLACE'
            ]
        ])


    def init_episode(self, index: int) -> List[str]:
        self.register_change_point_conditions([
            DetectedCondition(self.door, self.success_door),
            CustomConditionSet([
                DetectedCondition(self.tray, self.success_door),
                NothingGrasped(self.robot.gripper)
            ]),
            DetectedCondition(self.plate, self.success_detector, negated=True),
            DetectedCondition(self.plate, self.success_plate)
        ])
        return ['empty the dishwasher', 'take dishes out of dishwasher',
                'open the  dishwasher door, slide the rack out and remove the '
                'dishes']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 2.], [0, 0, 3.14 / 2.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
