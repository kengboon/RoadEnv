import random

class ReplayBuffer:
    def __init__(self, maxlen=None):
        self._buffer = []
        self.maxlen = maxlen

    def append(self, elem):
        self._buffer.append(elem)
        if self.maxlen is not None and len(self) > self.maxlen:
            self._buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self._buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self._buffer)