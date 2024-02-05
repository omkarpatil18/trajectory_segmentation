import numpy as np
from collections import Counter

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
        
        counter = Counter (self.change_point)
        x = len (self.instructions[0])
        extra_len = counter[x]
        for i in range (len(self.change_point) - 1, extra_len - 1, -1):
            self.change_point[i] = self.change_point[i - extra_len]

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)
    
    def get_chunk(self, query, start, end):
        self._observations = self._observations[start: end + 1]
        self.change_point = self.change_point[start: end + 1]
        index = -1
        for instructions in self.instructions:
            for i, instruction in enumerate (instructions):
                if instruction == query:
                    index = i
                    break
        self.instructions = [[instructions[index]] for instructions in self.instructions]