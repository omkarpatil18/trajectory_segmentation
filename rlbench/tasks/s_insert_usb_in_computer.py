from typing import List, Tuple
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped



class SInsertUsbInComputer(Task):

    def init_task(self) -> None:
        success_sensor = ProximitySensor('success')
        negate = ProximitySensor('negate')
        usb = Shape('usb')
        usb_tip = Shape('tip')
        self.register_graspable_objects([usb])
        self.register_success_conditions([
            DetectedCondition(usb, negate, negated = True),
            DetectedCondition(usb_tip, success_sensor)])
        
        self.register_change_point_conditions([
            DetectedCondition(usb, negate, negated = True),
            DetectedCondition(usb_tip, success_sensor)
        ])
        self.register_instructions([
            [
                'Retrieve the USB from the cuboid.',
                'Insert the USB into the computer.'
            ],
            [
                'Pick up the USB from the cuboid.',
                'Connect the USB to the computer.'
            ],
            [
                'Take the USB out of the cuboid.',
                'Plug the USB into the computer.'
            ],
            [
                'Gather the USB from the cuboid.',
                'Insert the USB securely into the computer.'
            ],
            [
                'Remove the USB from the cuboid.',
                'Place the USB into the computer\'s port.'
            ],
            [
                'SKILL_PICK',
                'SKILL_INSERT'
            ]
        ])

    def init_episode(self, index: int) -> List[str]:
        return ['insert usb in computer',
                'pick up the usb and put it in the computer',
                'slide the usb into the usb slot',
                'insert the usb stick into the usb port']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
