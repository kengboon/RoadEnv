from collections import deque
import random

class ReplayBuffer:
    def __init__(self, maxlen=None):
        self._buffer = deque([], maxlen=maxlen)

    def append(self, elem):
        self._buffer.append(elem)

    def sample(self, batch_size):
        batch = random.sample(self._buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self._buffer)