from typing import List
import numpy as np
from pyrep.objects import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors


class SSquarePeg(Task):

    def init_task(self) -> None:
        self._square_ring0 = Shape('square_ring0')
        self._square_ring1 = Shape('square_ring1')
        self._square_ring2 = Shape('square_ring2')
        
        self._success_centre0 = Dummy('success_centre0')
        self._success_centre1 = Dummy('success_centre1')
        self._success_centre2 = Dummy('success_centre2')

        
        success_detectors0 = [ProximitySensor(
            'success_detector%d' % i) for i in range(4)]
        success_detectors1 = [ProximitySensor(
            'success_detector%d' % i) for i in range(4, 8)]
        success_detectors2 = [ProximitySensor(
            'success_detector%d' % i) for i in range(8, 12)]
        
        self.register_graspable_objects([
            self._square_ring0, self._square_ring1, self._square_ring2])

        success_condition0 = ConditionSet([DetectedCondition(
            self._square_ring0, sd) for sd in success_detectors0])
        success_condition1 = ConditionSet([DetectedCondition(
            self._square_ring1, sd) for sd in success_detectors1])
        success_condition2 = ConditionSet([DetectedCondition(
            self._square_ring2, sd) for sd in success_detectors2])
        
        self.register_success_conditions([success_condition0, 
            success_condition1, success_condition2])
        self.register_change_point_conditions([
            success_condition0, success_condition1, success_condition2
        ])

    def init_episode(self, index: int) -> List[str]:
        color_choice = np.random.choice(list(range(len(colors))),
            size=6, replace=False)

        spokes = [Shape('pillar0'), Shape('pillar1'), Shape('pillar2')]
        chosen_pillar = np.random.choice(
            list(range(3)), size = 3, replace = True)
        color_names = []

        name, rgb = colors[0]
        spokes[chosen_pillar[0]].set_color(rgb)
        color_names.append (name)
        _, _, z = self._success_centre0.get_position()
        x, y, _ = spokes[chosen_pillar[0]].get_position()
        self._success_centre0.set_position([x, y, z])

        name, rgb = colors[1]
        spokes[chosen_pillar[1]].set_color(rgb)
        color_names.append (name)
        _, _, z = self._success_centre1.get_position()
        x, y, _ = spokes[chosen_pillar[1]].get_position()
        self._success_centre1.set_position([x, y, z])

        name, rgb = colors[2]
        spokes[chosen_pillar[2]].set_color(rgb)
        color_names.append (name)
        _, _, z = self._success_centre2.get_position()
        x, y, _ = spokes[chosen_pillar[2]].get_position()
        self._success_centre2.set_position([x, y, z])

        self._square_ring0.set_color(colors[color_choice[3]][1])
        color_names.append (colors[color_choice[3]][0])
        self._square_ring1.set_color(colors[color_choice[4]][1])
        color_names.append (colors[color_choice[4]][0])
        self._square_ring2.set_color(colors[color_choice[5]][1])
        color_names.append (colors[color_choice[5]][0])

        b0 = SpawnBoundary([Shape('boundary0')])
        b1 = SpawnBoundary([Shape('boundary4')])
        b2 = SpawnBoundary([Shape('boundary5')])

        b0.sample(self._square_ring0)
        b1.sample(self._square_ring1)
        b2.sample(self._square_ring2)

        self.register_instructions([
            [
                'Place the %s square ring onto the %s spoke.' % (color_names[3], color_names[0]),
                'Put the %s square ring on the %s spoke.' % (color_names[4], color_names[1]),
                'Position the %s square ring on the %s spoke.' % (color_names[5], color_names[2])
            ],
            [
                'Put the %s square ring on the %s spoke.' % (color_names[3], color_names[0]),
                'Position the %s square ring onto the %s spoke.' % (color_names[4], color_names[1]),
                'Place the %s square ring onto the %s spoke.' % (color_names[5], color_names[2])
            ],
            [
                'Place the %s square ring on the %s spoke.' % (color_names[3], color_names[0]),
                'Position the %s square ring on the %s spoke.' % (color_names[4], color_names[1]),
                'Put the %s square ring onto the %s spoke.' % (color_names[5], color_names[2])
            ],
            [
                'Put the %s square ring onto the %s spoke.' % (color_names[3], color_names[0]),
                'Position the %s square ring on the %s spoke.' % (color_names[4], color_names[1]),
                'Place the %s square ring on the %s spoke.' % (color_names[5], color_names[2])
            ],
            [
                'Position the %s square ring on the %s spoke.' % (color_names[3], color_names[0]),
                'Put the %s square ring onto the %s spoke.' % (color_names[4], color_names[1]),
                'Place the %s square ring on the %s spoke.' % (color_names[5], color_names[2])
            ]
        ])

        return ['put the ring on the %s spoke' % color_names[0],
                'slide the ring onto the %s colored spoke' % color_names[0],
                'place the ring onto the %s spoke' % color_names[0]]

    def variation_count(self) -> int:
        return 1
