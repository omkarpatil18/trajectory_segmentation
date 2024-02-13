from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, \
    OrConditions
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors


class SPlaceCups(Task):

    def init_task(self) -> None:
        self._cups = [Shape('mug%d' % i) for i in range(3)]
        self._spokes = [Shape('place_cups_holder_spoke%d' % i) for i in
                        range(3)]
        self._cups_boundary = Shape('mug_boundary')
        self._negate = ProximitySensor('negate')
        self._w1 = Dummy('waypoint1')
        self._w4 = Dummy('waypoint4')
        success_detectors = [
            ProximitySensor('success_detector%d' % i) for i in range(3)]
        self._on_peg_conditions = [OrConditions([
            DetectedCondition(self._cups[ci], success_detectors[sdi]) for sdi in
            range(3)]) for ci in range(3)]
        self._picked_up_conditions = [
            DetectedCondition(self._cups[ci], self._negate, negated = True)
            for ci in range(3)
        ]
        self.register_graspable_objects(self._cups)
        self._initial_relative_cup = self._w1.get_pose(self._cups[0])
        self._initial_relative_spoke = self._w4.get_pose(self._spokes[0])

    def init_episode(self, index: int) -> List[str]:
        index = 2
        self._cups_placed = 0
        self._index = index
        
        color_choice = np.random.choice(list(range(len(colors))),
            size=3, replace=False)

        color_names = []
        for i in range(3):
            name, rgb = colors[color_choice[i]]
            self._spokes[i].set_color(rgb)
            color_names.append (name)

        b = SpawnBoundary([self._cups_boundary])
        [b.sample(c, min_distance=0.10) for c in self._cups]
        success_conditions = [NothingGrasped(self.robot.gripper)
                              ] + self._on_peg_conditions[:index + 1] + self._picked_up_conditions
        self.register_success_conditions(success_conditions)
        self.register_change_point_conditions([
            self._picked_up_conditions[0],
            self._on_peg_conditions[0],
            self._picked_up_conditions[1],
            self._on_peg_conditions[1],
            self._picked_up_conditions[2],
            self._on_peg_conditions[2]
        ])

        self.register_instructions([
            [
                "Retrieve a cup from the table.",
                "Position the cup on the %s spoke of the cup holder." % color_names[0],
                "Take another cup from the table.",
                "Place this cup on the %s spoke of the cup holder." % color_names[1],
                "Collect one more cup from the table.",
                "Set this cup on the %s spoke of the cup holder." % color_names[2]
            ],
            [
                "Pick up a cup from the table.",
                "Put the cup on the %s spoke of the cup holder." % color_names[0],
                "Take a cup from the table.",
                "Position it on the %s spoke of the cup holder." % color_names[1],
                "Gather another cup from the table.",
                "Place it on the %s spoke of the cup holder." % color_names[2]
            ],
            [
                "Get a cup from the table.",
                "Set the cup on the %s spoke of the cup holder." % color_names[0],
                "Pick up one more cup from the table.",
                "Position it on the %s spoke of the cup holder." % color_names[1],
                "Take an additional cup from the table.",
                "Place it on the %s spoke of the cup holder." % color_names[2]
            ],
            [
                "Collect a cup from the table.",
                "Arrange the cup on the %s spoke of the cup holder." % color_names[0],
                "Grab another cup from the table.",
                "Position it on the %s spoke of the cup holder." % color_names[1],
                "Take one more cup from the table.",
                "Place it on the %s spoke of the cup holder." % color_names[2]
            ],
            [
                "Pick up a cup from the table.",
                "Place the cup on the %s spoke of the cup holder." % color_names[0],
                "Retrieve another cup from the table.",
                "Position it on the %s spoke of the cup holder." % color_names[1],
                "Take one more cup from the table.",
                "Set it on the %s spoke of the cup holder." % color_names[2]
            ],
            [
                'SKILL_PICK_CUP',
                'SKILL_PLACE_%s' % color_names[0],
                'SKILL_PICK_CUP',
                'SKILL_PLACE_%s' % color_names[1],
                'SKILL_PICK_CUP',
                'SKILL_PLACE_%s' % color_names[2]
            ]
        ])
        
        self.register_waypoint_ability_start(
            0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)


        
        return ['place %d cups on the cup holder' % (index + 1),
                    'pick up %d cups and place them on the holder'
                    % (index + 1),
                    'move %d cups from the table to the mug tree'
                    % (index + 1),
                    'pick up %d mugs and slide their handles onto the cup '
                    'holder spokes' % (index + 1)]

    def variation_count(self) -> int:
        return 1

    def _move_above_next_target(self, waypoint):
        self._w1.set_parent(self._cups[self._cups_placed])
        self._w4.set_pose(
            self._initial_relative_spoke,
            relative_to=self._spokes[self._cups_placed])
        self._w1.set_pose(
            self._initial_relative_cup,
            relative_to=self._cups[self._cups_placed])
        self._cups_placed += 1

    def _repeat(self):
        return self._cups_placed < self._index + 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -np.pi / 2], [0.0, 0.0, np.pi / 2]
