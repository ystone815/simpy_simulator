
import simpy

class IndexAllocator:
    def __init__(self, env, num_indices):
        self.env = env
        self.indices = simpy.Store(env, capacity=num_indices)
        self.indices.items = list(range(num_indices))

    def get(self):
        return self.indices.get()

    def put(self, index):
        return self.indices.put(index)
