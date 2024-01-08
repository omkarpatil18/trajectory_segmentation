from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy

from rlbench.backend.conditions import DetectedCondition, NothingGrasped, ConditionSet
from rlbench.backend.task import Task

from typing import List

import numpy as np
import random

COLORS = ['white', 'black']
PIECES = ['king', 'kingside_bishop', 'kingside_rook', 'kingside_knight', 'pawn_a',
          'pawn_b', 'pawn_c', 'pawn_d', 'pawn_e', 'pawn_f', 'pawn_g', 'pawn_h',
          'queen', 'queenside_bishop', 'queenside_knight', 'queenside_rook']

class SSetupChess(Task):
    MAX_DISPLACEMENTS = 6

    def init_task(self) -> None:
        self.board = Shape('chess_board_base')
        self.pieces = [Shape(f'{color}_{piece}_dynamic') for color in COLORS for piece in PIECES]

        self.positions = [piece.get_position(self.board) for piece in self.pieces]
        self.rotations = [piece.get_orientation(self.board) for piece in self.pieces]

        self.success_detectors = [ProximitySensor(f'detector_{file}{rank}') for file in 'abcdefgh' for rank in [1, 2, 7, 8]]
        self.success_conditions = [NothingGrasped(self.robot.gripper)]
        
        self.change_point_conditions = []

        for piece, detector in zip(self.pieces, self.success_detectors):
            x, y, z = piece.get_position(self.board)
            z = detector.get_position(self.board)[2]
            detector.set_position((x, y, z), self.board)
            self.success_conditions.append(DetectedCondition(piece, detector))
            self.change_point_conditions.append (DetectedCondition(piece, detector))

        self.register_success_conditions(self.success_conditions)
        self.register_graspable_objects(self.pieces)
        self.register_change_point_conditions(self.change_point_conditions)

    def init_episode(self, index: int) -> List[str]:
        
        for piece, position, rotation in zip(self.pieces, self.positions, self.rotations):
            piece.set_position(position, self.board)
            piece.set_orientation(rotation, self.board)

        self.nsetup = 1 + index % self.MAX_DISPLACEMENTS
        self.placed = -1
        self.places = random.sample([(dx, dy) for dx in range(8) for dy in range(1, 3)], self.nsetup)

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

        self.indices = np.random.choice(np.arange(len(self.pieces)), self.nsetup, replace = False)
        self.targets = [self.pieces[idx] for idx in self.indices]

        delta = 4.7888e-2

        for piece, (x, y) in zip(self.targets, self.places):
            x = -1.6759e-1 + x * delta
            y = -7.1825e-2 + y * delta
            z = piece.get_position(self.board)[2]
            piece.set_position([x, y, z], self.board, reset_dynamics = False)

        if self.nsetup == 1:
            cmds = ['place the remaining chess piece in its initial position on the board']
        else:
            cmds = [f'place the {self.nsetup} remaining chess pieces in their initial positions on the board']

        cmds.extend([
            'prepare the chessboard',
            'get the chessboard ready for a game of chess',
            'setup the chessboard',
            'setup chess',
            'arrange the chess pieces ready for a game',
            'get chess pieces ready'
        ])
        self.instructions = []
        for i in range(self.nsetup):
            piece = self.targets[i]
            self.instructions.append (self._get_instructions(piece.get_name()))
        self.register_instructions([self.instructions])

        return cmds

    def variation_count(self) -> int:
        return self.MAX_DISPLACEMENTS

    def _move_above_next_target(self, waypoint):
        self.placed += 1
        index = self.indices[self.placed]
        piece = self.targets[self.placed]
        final = self.positions[index]
        
        w1 = Dummy("waypoint1")
        w4 = Dummy("waypoint4")

        xt, yt, zt = piece.get_position(self.board) # current position
        xf, yf, zf = final                          # target position

        w1.set_position([xt, yt, zt], self.board, reset_dynamics = False)
        w4.set_position([xf - 1e-3, yf, zf + 0.1], self.board, reset_dynamics = False)

    def _repeat(self):
        return self.placed + 1 < self.nsetup

    def _get_instructions (self, name):
        name = name.replace ('_dynamic', '')
        name = name.split ('_')
        color = name[0]

        if name[1] == 'pawn':
            piece = name[1]
            if name[2] == 'a':
                position = 'by the queen side in front of %s rook' % color
            elif name[2] == 'b':
                position = 'by the queen side in front of %s knight' % color
            elif name[2] == 'c':
                position = 'by the queen side in front of %s bishop' % color
            elif name[2] == 'd':
                position = 'in front of %s queen' % color
            elif name[2] == 'e':
                position = 'in front of %s king' % color
            elif name[2] == 'h':
                position = 'by the king side in front of %s rook' % color
            elif name[2] == 'g':
                position = 'by the king side in front of %s knight' % color
            elif name[2] == 'f':
                position = 'by the king side in front of %s bishop' % color
            
        elif name[1] in ['king', 'queen']:
            piece = name[1]
            if piece == 'king':
                position = 'in between queen and %s bishop' % color
            else:
                position = 'in between king and %s bishop' % color
        
        else:
            piece = name[2]
            if name[1] == 'kingside':
                position = 'on %s king side' % color
            else:
                position = 'on %s queen side' % color
        
        return f"Place the {color} {piece} {position}"
