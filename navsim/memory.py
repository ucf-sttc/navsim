from typing import Union, Optional
import numpy as np
import pickle
import sys
from navsim.util import sizeof_fmt, image_layout


class Memory:
    """
    Stores and returns list of numpy arrays for s a r s_ d

    assumes you are storing s a r s_ d : state action reward next_state done
    # state / next_state : tuple of n-dim numpy arrays
    # a : 1-d numpy array of floats or ints
    # r : 1-d numpy array of floats #TODO: rewards could also be n-dim
    # d : 1-d numpy array of booleans

    # TODO: if we made this tensor based instead of np, would it make it better ?
    """

    @staticmethod
    def sample_info(s, a, r, s_, d):
        print("Sample Info")
        print('-----------')
        print("Shapes:")
        print('s:', [a.shape for a in s])
        print('a:', a.shape)
        print('r:', r.shape)
        print('s_:', [a.shape for a in s_])
        print('d:', d.shape)

    def __init__(self, capacity: int,
                 state_shapes: Union[list, tuple, int],
                 action_shape: Union[list, tuple, int],
                 seed: Optional[float] = None):
        """

        :param capacity: total capacity of the memory
        :param state_shapes: a tuple of state dimensions
        :param action_shape: int representing number of actions
        :param seed: seed to provide to numpy
        """
        self.capacity = capacity
        self.ptr: int = 0
        self.size: int = 0

        self.seed: Optional[float] = seed
        # from numpy.random import MT19937, RandomState, Generator
        # mt = MT19937(seed)
        # rs = RandomState(mt)
        # rg = Generator(mt)
        self.rng = np.random.default_rng(seed=seed)

        # if isinstance(state_shapes, (list, tuple)):  # state_dims is an int
        if isinstance(state_shapes, (int, float)):  # state_dims is an int
            state_shapes = [
                [state_shapes]]  # make state_dims a list of sequences
        elif isinstance(state_shapes[0],
                        (int, float)):  # state_dims first member is an int
            state_shapes = [state_shapes]  # make state_dims a list of sequences
        # else state_dims is a list of sequences, i.e. [[state_shape], [state_shap], [state_shape]]

        s_len = len(state_shapes)
        self.s = [None] * s_len
        self.s_ = [None] * s_len

        for i in range(s_len):
            state_shapes[i] = [capacity] + list(state_shapes[i])
            self.s[i] = np.full(state_shapes[i], 0.0)
            self.s_[i] = np.full(state_shapes[i], 0.0)

        action_shape = [capacity] + list(action_shape)
        self.a = np.full(action_shape, 0.0)

        self.r = np.full((capacity, 1), 0.0)
        self.d = np.full((capacity, 1), False)

    def append(self, s: Union[list, tuple, int, float],
               a: Union[list, tuple, int, float],
               r: float,
               s_: Union[list, tuple, int, float],
               d: int):
        if isinstance(s, (int, float)):  # state is an int
            s = [[s]]  # make state a list of sequences
            s_ = [[s_]]
        elif isinstance(s[0], (int, float)):  # state does not have multiple seq
            s = [s]
            s_ = [s_]
        for i in range(len(s)):
            self.s[i][self.ptr] = s[i]
            self.s_[i][self.ptr] = s_[i]

        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size):
        if (size == 0) or (size >= self.size):
            # idx = list(range(self.size))   # just return all
            return self.s, self.a, self.r, self.s_, self.d
        else:
            idx = self.rng.integers(low=0, high=self.size, size=size)
            s_len = len(self.s)
            return [self.s[i][idx] for i in range(s_len)], \
                self.a[idx], \
                self.r[idx], \
                [self.s_[i][idx] for i in range(s_len)], \
                self.d[idx]

    def info(self):
        print('-----------')
        print("Memory Info")
        print('-----------')
        print('capacity:', self.capacity)
        print('size:', self.size)
        print('seed:', self.seed)
        print("Shapes:")
        print('s:', [a.shape for a in self.s])
        print('a:', self.a.shape)
        print('r:', self.r.shape)
        print('s_:', [a.shape for a in self.s_])
        print('d:', self.d.shape)
        print("Sizes in bytes:")
        print('memory:', sizeof_fmt(sys.getsizeof(self)))
        print('s:', sizeof_fmt(sys.getsizeof(self.s)))
        print('a:', sizeof_fmt(sys.getsizeof(self.a)))
        print('r:', sizeof_fmt(sys.getsizeof(self.r)))
        print('s_:', sizeof_fmt(sys.getsizeof(self.s_)))
        print('d:', sizeof_fmt(sys.getsizeof(self.d)))

    def __len__(self):
        return len(self.mem)

    def __repr__(self):
        return str(self.mem)

    def __str__(self):
        return f'capacity={self.capacity} \nlength={len(self.mem)} \ncurrent pointer={self.ptr} \nmemory contents:\n{self.mem}'

    def __getitem__(self, sliced):
        return Memory(capacity=self.capacity, seed=self.seed,
                      mem=self.mem[sliced])

    def save_to_pkl(self, filename):
        with open(filename, "wb") as out_file:
            pickle.dump(self, out_file)

    def save_to_npz(self, filename):
        pass

    @staticmethod
    def load_from_pkl(filename):
        with open(filename, "rb") as in_file:
            return pickle.load(in_file)

    def pretty_print(self, n_rows=0):
        """

        :param n_rows: 0 means all
        :return:
        """
        pass
