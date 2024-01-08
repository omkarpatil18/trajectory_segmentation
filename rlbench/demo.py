import numpy as np


class Demo(object):

    def __init__(self, observations, random_seed=None, num_reset_attempts = None):
        self._observations = observations
        self.random_seed = random_seed
        self.num_reset_attempts = num_reset_attempts
        self.instructions = observations[0].instruction
        self.change_point = [sum(bool(x) for x in obs.success_state) for obs in observations]
        
        min_element = self.change_point[-1]
        for i in range (len(self.change_point) - 1, -1, -1):
            min_element = min (min_element, self.change_point[i])
            self.change_point[i] = min_element
        self.change_point = [x - min_element for x in self.change_point]

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)
