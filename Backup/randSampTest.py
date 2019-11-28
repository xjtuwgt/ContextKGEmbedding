import random
import bisect
import numpy as np
from time import time

class WeightedRandomGenerator(object):
    import random
    import bisect
    def __init__(self, weights):
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)

    def __call__(self):
        return self.next()

elements = list(range(10))
probability = np.array([1.0]*10)*1.0#np.random.random(10)

weight_sampler = WeightedRandomGenerator(probability)
start = time()
for i in range(10000):
    weight_sampler.next()
print(time() - start)