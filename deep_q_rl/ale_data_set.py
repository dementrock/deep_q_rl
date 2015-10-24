"""This class stores all of the samples for training.  It is able to
construct randomly selected batches of phi's from the stored history.
"""

import numpy as np
import time
import theano

floatX = theano.config.floatX

class DataSet(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, ram_size, rng, max_steps=1000):
        """Construct a DataSet.

        Arguments:
            width, height - image size
            max_steps - the number of time steps to store
            phi_length - number of images to concatenate into a state
            rng - initialized numpy random number generator, used to
            choose random minibatches

        """
        # TODO: Specify capacity in number of state transitions, not
        # number of saved time steps.

        # Store arguments.
        self.ram_size = ram_size
        self.max_steps = max_steps
        self.rng = rng

        # Allocate the circular buffers and indices.
        self.rams = np.zeros((max_steps, ram_size), dtype=floatX)
        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminal = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0

    def add_sample(self, ram, action, reward, terminal):
        """Add a time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.rams[self.top] = ram
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminal[self.top] = terminal

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
        self.top = (self.top + 1) % self.max_steps

    def __len__(self):
        """Return an approximate count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return max(0, self.size - 1)

    def last_ram(self):
        return self.rams[self.top - 1]

    #def last_phi(self):
    #    """Return the most recent phi (sequence of image frames)."""
    #    indexes = np.arange(self.top - self.phi_length, self.top)
    #    return self.imgs.take(indexes, axis=0, mode='wrap')

    #def phi(self, img):
    #    """Return a phi (sequence of image frames), using the last phi_length -
    #    1, plus img.

    #    """
    #    indexes = np.arange(self.top - self.phi_length + 1, self.top)

    #    phi = np.empty((self.phi_length, self.height, self.width), dtype=floatX)
    #    phi[0:self.phi_length - 1] = self.imgs.take(indexes,
    #                                                axis=0,
    #                                                mode='wrap')
    #    phi[-1] = img
    #    return phi

    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.

        """
        # Allocate the response.
        states = np.zeros((batch_size,
                           self.ram_size), dtype=floatX)
        actions = np.zeros((batch_size, 1), dtype='int32')
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')
        next_states = np.zeros((batch_size,
                                self.ram_size), dtype=floatX)

        count = 0
        while count < batch_size:
            # Randomly choose a time step from the replay memory.
            index = self.rng.randint(self.bottom,
                                     self.bottom + self.size - 1)

            # Add the state transition to the response.
            states[count] = self.rams[index % self.size]#.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions[index % self.size]
            rewards[count] = self.rewards[index % self.size]
            terminal[count] = self.terminal[index % self.size]#.take(end_index, mode='wrap')
            next_states[count] = self.rams[(index + 1) % self.size]
            count += 1

        return states, actions, rewards, next_states, terminal


# TESTING CODE BELOW THIS POINT...

def simple_tests():
    np.random.seed(222)
    dataset = DataSet(ram_size=10,
                      rng=np.random.RandomState(42),
                      max_steps=6)
    for i in range(10):
        ram = np.random.randint(0, 256, size=(10,))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        print 'ram', ram
        dataset.add_sample(ram, action, reward, terminal)
        print "I", dataset.rams
        print "A", dataset.actions
        print "R", dataset.rewards
        print "T", dataset.terminal
        print "SIZE", dataset.size
        print
    print "LAST RAM", dataset.last_ram()
    print
    print 'BATCH', dataset.random_batch(2)


def speed_tests():

    dataset = DataSet(ram_size=128,
                      rng=np.random.RandomState(42),
                      max_steps=20000)

    ram = np.random.randint(0, 256, size=(128,))
    action = np.random.randint(16)
    reward = np.random.random()
    start = time.time()
    for i in range(100000):
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset.add_sample(ram, action, reward, terminal)
    print "samples per second: ", 100000 / (time.time() - start)

    start = time.time()
    for i in range(200):
        a = dataset.random_batch(32)
    print "batches per second: ", 200 / (time.time() - start)


def trivial_tests():

    dataset = DataSet(width=2, height=1,
                      rng=np.random.RandomState(42),
                      max_steps=3, phi_length=2)

    img1 = np.array([[1, 1]], dtype='uint8')
    img2 = np.array([[2, 2]], dtype='uint8')
    img3 = np.array([[3, 3]], dtype='uint8')

    dataset.add_sample(img1, 1, 1, False)
    dataset.add_sample(img2, 2, 2, False)
    dataset.add_sample(img3, 2, 2, True)
    print "last", dataset.last_ram()
    print "random", dataset.random_batch(1)


def max_size_tests():
    dataset1 = DataSet(ram_size=12,
                      rng=np.random.RandomState(42),
                      max_steps=10)
    dataset2 = DataSet(ram_size=12,
                      rng=np.random.RandomState(42),
                      max_steps=1000)
    for i in range(100):
        img = np.random.randint(0, 256, size=(12,))
        action = np.random.randint(16)
        reward = np.random.random()
        terminal = False
        if np.random.random() < .05:
            terminal = True
        dataset1.add_sample(img, action, reward, terminal)
        dataset2.add_sample(img, action, reward, terminal)
        np.testing.assert_array_almost_equal(dataset1.last_ram(),
                                             dataset2.last_ram())
        print "passed"


def test_memory_usage_ok():
    import memory_profiler
    dataset = DataSet(ram_size=128,
                      rng=np.random.RandomState(42),
                      max_steps=100000)
    last = time.time()

    for i in xrange(1000000000):
        if (i % 100000) == 0:
            print i
        dataset.add_sample(np.random.random((128,)), 1, 1, False)
        if i > 200000:
            states, actions, rewards, next_states, terminals = \
                                        dataset.random_batch(32)
        if (i % 10007) == 0:
            print time.time() - last
            mem_usage = memory_profiler.memory_usage(-1)
            print len(dataset), mem_usage
        last = time.time()

def main():
    speed_tests()
    test_memory_usage_ok()
    max_size_tests()
    simple_tests()

if __name__ == "__main__":
    main()
