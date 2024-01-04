import numpy as np
from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition, ConditionSet
from rlbench.const import colors

SHAPE_NAMES = ['cube', 'cylinder', 'triangular prism', 'star', 'moon']


class SShapeSorter(Task):

    def init_task(self) -> None:
        self.shape_sorter = Shape('shape_sorter')
        self.success_sensor = ProximitySensor('success')
        self.indexes = np.random.choice (list(range(len(SHAPE_NAMES))), 
            len(SHAPE_NAMES), replace = False)

        self.shapes = [Shape(ob.replace(' ', '_')) for ob in SHAPE_NAMES]
        self.drop_points = [
            Dummy('%s_drop_point' % ob.replace(' ', '_'))
            for ob in SHAPE_NAMES]
        self.grasp_points = [
            Dummy('%s_grasp_point' % ob.replace(' ', '_'))
            for ob in SHAPE_NAMES]
        
        self.waypoint1 = Dummy('waypoint1')
        self.waypoint4 = Dummy('waypoint4')
        self.waypoint6 = Dummy('waypoint6')
        self.waypoint9 = Dummy('waypoint9')
        self.waypoint11 = Dummy('waypoint11')
        self.waypoint14 = Dummy('waypoint14')
        self.waypoint16 = Dummy('waypoint16')
        self.waypoint19 = Dummy('waypoint19')
        self.waypoint21 = Dummy('waypoint21')
        self.waypoint24 = Dummy('waypoint24')

        self.register_graspable_objects(self.shapes)

        self.register_waypoint_ability_start(0, self._set_grasp)
        self.register_waypoint_ability_start(3, self._set_drop)
        self.boundary = SpawnBoundary([Shape('boundary')])

    def init_episode(self, index) -> List[str]:
        
        color_choice = np.random.choice(list(range(len(colors))),
            size=5, replace=False)

        color_names = []
        for i in range(5):
            index = self.indexes[i]
            obj = self.shapes[index]
            name, rgb = colors[color_choice[i]]
            color_names.append (name)
            obj.set_color(rgb)

        self.variation_index = index
        shape = SHAPE_NAMES[index]

        cond_set = ConditionSet([
            DetectedCondition(self.shapes[self.indexes[0]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[1]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[2]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[3]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[4]], self.success_sensor),
        ])
        self.register_success_conditions([cond_set])

        self.register_change_point_conditions([
            DetectedCondition(self.shapes[self.indexes[0]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[1]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[2]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[3]], self.success_sensor),
            DetectedCondition(self.shapes[self.indexes[4]], self.success_sensor),
        ])

        self.register_instructions([
            ['Put the %s %s block in the shape sorter' % (color_names[0], SHAPE_NAMES[self.indexes[0]]),
            'Put the %s %s block in the shape sorter' % (color_names[1], SHAPE_NAMES[self.indexes[1]]),
            'Put the %s %s block in the shape sorter' % (color_names[2], SHAPE_NAMES[self.indexes[2]]),
            'Put the %s %s block in the shape sorter' % (color_names[3], SHAPE_NAMES[self.indexes[3]]),
            'Put the %s %s block in the shape sorter' % (color_names[4], SHAPE_NAMES[self.indexes[4]])
            ]
        ])

        self.boundary.clear()
        [self.boundary.sample(s, min_distance=0.05) for s in self.shapes]

        return ['put the %s in the shape sorter' % shape,
                'pick up the %s and put it in the sorter' % shape,
                'place the %s into its slot in the shape sorter' % shape,
                'slot the %s into the shape sorter' % shape]

    def variation_count(self) -> int:
        return len(SHAPE_NAMES)

    def _set_grasp(self, _):
        gp = self.grasp_points[self.indexes[0]]
        self.waypoint1.set_pose(gp.get_pose())

        gp = self.grasp_points[self.indexes[1]]
        self.waypoint6.set_pose(gp.get_pose())

        gp = self.grasp_points[self.indexes[2]]
        self.waypoint11.set_pose(gp.get_pose())

        gp = self.grasp_points[self.indexes[3]]
        self.waypoint16.set_pose(gp.get_pose())

        gp = self.grasp_points[self.indexes[4]]
        self.waypoint21.set_pose(gp.get_pose())

    def _set_drop(self, _):
        dp = self.drop_points[self.indexes[0]]
        self.waypoint4.set_pose(dp.get_pose())

        dp = self.drop_points[self.indexes[1]]
        self.waypoint9.set_pose(dp.get_pose())

        dp = self.drop_points[self.indexes[2]]
        self.waypoint14.set_pose(dp.get_pose())

        dp = self.drop_points[self.indexes[3]]
        self.waypoint19.set_pose(dp.get_pose())

        dp = self.drop_points[self.indexes[4]]
        self.waypoint24.set_pose(dp.get_pose())
