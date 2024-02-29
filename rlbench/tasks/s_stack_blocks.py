from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedSeveralCondition,
    DetectedCondition,
    CustomDetectedCondition,
    CustomConditionSet,
)
from rlbench.backend.conditions import NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors

MAX_STACKED_BLOCKS = 2
DISTRACTORS = 0


class SStackBlocks(Task):

    def init_task(self) -> None:
        self.blocks_stacked = 0
        self.target_blocks = [Shape("stack_blocks_target%d" % i) for i in range(2)]
        # self.distractors = [
        #    Shape("stack_blocks_distractor%d" % i) for i in range(DISTRACTORS)
        # ]

        self.boundaries = [Shape("stack_blocks_boundary%d" % i) for i in range(2)]

        self.register_graspable_objects(self.target_blocks)

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoint_ability_start(3, self._move_above_drop_zone)
        self.register_waypoint_ability_start(5, self._is_last)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        # For each color, we want to have 2, 3 or 4 blocks stacked
        color_index = int(index / MAX_STACKED_BLOCKS)
        self.blocks_to_stack = MAX_STACKED_BLOCKS  # 2 + index % MAX_STACKED_BLOCKS

        color_name = " "
        color_choice = np.random.choice(list(range(len(colors))), size=8, replace=False)
        color_choice = [0, 3, 4, 6, 9, 11, 13, 15]

        color_names = []
        for i in range(2):
            name, rgb = colors[color_choice[i]]
            color_names.append(name)
            obj = self.target_blocks[i]
            obj.set_color(rgb)

        success_detector = ProximitySensor("stack_blocks_success")
        negate = ProximitySensor("negate")
        self.register_success_conditions(
            [
                DetectedSeveralCondition(
                    self.target_blocks, success_detector, self.blocks_to_stack
                ),
                NothingGrasped(self.robot.gripper),
            ]
        )

        self.register_change_point_conditions(
            [
                CustomDetectedCondition(self.target_blocks[0], negate, negated=True),
                CustomConditionSet(
                    [
                        DetectedCondition(self.target_blocks[0], success_detector),
                        NothingGrasped(self.robot.gripper),
                    ]
                ),
                CustomDetectedCondition(self.target_blocks[1], negate, negated=True),
                CustomConditionSet(
                    [
                        DetectedCondition(self.target_blocks[1], success_detector),
                        # NothingGrasped(self.robot.gripper),
                    ]
                ),
            ]
        )

        self.register_instructions(
            [
                [  # changed for behavior cloning experiments
                    "Pick up the %s block." % color_names[0],
                    "Place the %s block at the green center." % color_names[0],
                    "Pick up the %s block." % color_names[1],
                    "Place the %s block on top of the %s block."
                    % (color_names[1], color_names[0]),
                ],
                [
                    "Take the %s block." % color_names[0],
                    "Put the %s block at the green center." % color_names[0],
                    "Retrieve the %s block." % color_names[1],
                    "Stack the %s block on top of the %s block."
                    % (color_names[1], color_names[0]),
                ],
                [
                    "Pick up the %s block." % color_names[0],
                    "Position the %s block at the green center." % color_names[0],
                    "Take the %s block." % color_names[1],
                    "Stack the %s block on top of the %s block."
                    % (color_names[1], color_names[0]),
                ],
                [
                    "Retrieve the %s block." % color_names[0],
                    "Place the %s block at the green center." % color_names[0],
                    "Grab the %s block." % color_names[1],
                    "Stack the %s block on top of the %s block."
                    % (color_names[1], color_names[0]),
                ],
                [
                    "Pick up the %s block." % color_names[0],
                    "Set the %s block at the green center." % color_names[0],
                    "Grab the %s block." % color_names[1],
                    "Stack the %s block on top of the %s block."
                    % (color_names[1], color_names[0]),
                ],
                [
                    "SKILL_PICK_%s" % color_names[0],
                    "SKILL_PLACE_ON_green",
                    "SKILL_PICK_%s" % color_names[1],
                    "SKILL_PLACE_ON_%s" % color_names[0],
                ],
            ]
        )

        self.blocks_stacked = 0

        # for i in range(4):
        #    name, rgb = colors[color_choice[i + 4]]
        #    obj = self.distractors[i]
        #    obj.set_color(rgb)
        z = np.random.choice(list(range(2)), size=2, replace=False)

        b1 = SpawnBoundary([self.boundaries[0]])
        for block in [self.target_blocks[z[0]]]:
            b1.sample(block, min_distance=0.1)

        b2 = SpawnBoundary([self.boundaries[1]])
        for block in [self.target_blocks[z[1]]]:
            b2.sample(block, min_distance=0.1)

        return [
            "stack %d %s blocks" % (self.blocks_to_stack, color_name),
            "place %d of the %s cubes on top of each other"
            % (self.blocks_to_stack, color_name),
            "pick up and set down %d %s blocks on top of each other"
            % (self.blocks_to_stack, color_name),
            "build a tall tower out of %d %s cubes"
            % (self.blocks_to_stack, color_name),
            "arrange %d %s blocks in a vertical stack on the table top"
            % (self.blocks_to_stack, color_name),
            "set %d %s cubes on top of each other" % (self.blocks_to_stack, color_name),
        ]

    def variation_count(self) -> int:
        return 1

    def _move_above_next_target(self, _):
        if self.blocks_stacked >= self.blocks_to_stack:
            raise RuntimeError("Should not be here.")
        w2 = Dummy("waypoint1")
        x, y, z = self.target_blocks[self.blocks_stacked].get_position()
        _, _, oz = self.target_blocks[self.blocks_stacked].get_orientation()
        ox, oy, _ = w2.get_orientation()
        w2.set_position([x, y, z])
        w2.set_orientation([ox, oy, -oz])

    def _move_above_drop_zone(self, waypoint):
        target = Shape("stack_blocks_target_plane")
        x, y, z = target.get_position()
        waypoint.get_waypoint_object().set_position(
            [x, y, z + 0.08 + 0.06 * self.blocks_stacked]
        )

    def _is_last(self, waypoint):
        last = self.blocks_stacked == self.blocks_to_stack - 1
        waypoint.skip = last

    def _repeat(self):
        self.blocks_stacked += 1
        return self.blocks_stacked < self.blocks_to_stack
