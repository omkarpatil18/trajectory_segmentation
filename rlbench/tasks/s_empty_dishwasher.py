from typing import List, Tuple
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition


class SEmptyDishwasher(Task):

    def init_task(self) -> None:
        success_detector = ProximitySensor('success')
        success_plate = ProximitySensor('success_plate')
        success_door = ProximitySensor('success_door')

        door = Shape('dishwasher_door')
        plate = Shape('dishwasher_plate')
        tray = Shape('dishwasher_tray')

        self.register_graspable_objects([plate])
        
        self.register_success_conditions(
            [DetectedCondition(door, success_door),
            DetectedCondition(tray, success_door),
            DetectedCondition(plate, success_detector, negated=True),
            DetectedCondition(plate, success_plate)])
        
        self.register_change_point_conditions([
            DetectedCondition(door, success_door),
            DetectedCondition(tray, success_door),
            DetectedCondition(plate, success_detector, negated=True),
            DetectedCondition(plate, success_plate)
        ])

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
            ]
        ])


    def init_episode(self, index: int) -> List[str]:
        return ['empty the dishwasher', 'take dishes out of dishwasher',
                'open the  dishwasher door, slide the rack out and remove the '
                'dishes']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 2.], [0, 0, 3.14 / 2.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
