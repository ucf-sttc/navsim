import pickle
from abc import ABC
from typing import Union, Optional

import cupy as cp
import numpy as np


class Memory(ABC):
    def __init__(self, capacity: int,
                 state_shapes: Union[list, tuple, int],
                 action_shape: Union[list, tuple, int],
                 seed: Optional[float] = None,
                 backend: object = np):
        self.capacity = capacity
        self.ptr: int = 0
        self.size: int = 0
        self.seed: Optional[float] = seed
        self.backend = backend
        self.a = self.r = self.d = None

        # if isinstance(state_shapes, (list, tuple)):  # state_dims is an int
        if isinstance(state_shapes, (int, float)):  # state_dims is an int
            state_shapes = [[state_shapes]]  # make it a list of sequences
        elif isinstance(state_shapes[0], (int, float)):
            # state_dims 1st member is an int
            state_shapes = [state_shapes]  # make it a list of sequences
        # else state_dims is a list of sequences,
        # i.e. [[state_shape], [state_shape], [state_shape]]

        self.s_len: int = len(state_shapes)
        self.s = [None] * self.s_len
        self.s_ = [None] * self.s_len

        self.rng = self.backend.random.default_rng(seed=seed)

        for i in range(self.s_len):
            s_shapes = [capacity] + list(state_shapes[i])
            self.s[i] = self.backend.full(s_shapes, 0.0)
            self.s_[i] = self.backend.full(s_shapes, 0.0)

        a_shape = [capacity] + list(action_shape)
        self.a = self.backend.full(a_shape, 0.0)

        self.r = self.backend.full((capacity, 1), 0.0)
        self.d = self.backend.full((capacity, 1), False)

    def __str__(self):
        return f'module: {type(self).__name__} \n' \
               f'capacity: {self.capacity} \n' \
               f'size: {self.size} \n' \
               f'current pointer: {self.ptr} \n' \
               f'seed: {self.seed} \n' \
               f'Shapes and types: \n' \
               f's: {[f"{i.shape} of {type(i)}" for i in self.s]} \n' \
               f'a: {self.a.shape} {type(self.a)} \n' \
               f'r: {self.r.shape} {type(self.r)} \n' \
               f's_: {[f"{i.shape} of {type(i)}" for i in self.s_]} \n' \
               f'd: {self.d.shape} {type(self.d)} \n'

    def info(self):
        print('-----------')
        print("Memory Info")
        print('-----------')
        print(self)
        print(f'----------------------------------------')

    @staticmethod
    def sample_info(s, a, r, s_, d):
        print('-----------')
        print("Sample Info")
        print('-----------')
        print("Shapes and types:")
        print('s:', [f"{i.shape} of {type(i)}" for i in s])
        print('a:', a.shape, type(a))
        print('r:', r.shape, type(r))
        print('s_:', [f"{i.shape} of {type(i)}" for i in s_])
        print('d:', d.shape, type(d))
        print('-----------')

    def sample(self, size):
        if (size == 0) or (size >= self.size):
            # idx = list(range(self.size))   # just return all
            return self.s, self.a, self.r, self.s_, self.d
        else:
            idx = self.rng.integers(low=0, high=self.size, size=size)
            return [self.s[i][idx] for i in range(self.s_len)], \
                   self.a[idx], \
                   self.r[idx], \
                   [self.s_[i][idx] for i in range(self.s_len)], \
                   self.d[idx]

    # def __len__(self):
    #    return len(self.mem)

    # def __repr__(self):
    #    return str(self.mem)

    # def __getitem__(self, sliced):
    #    return CupyMemory(capacity=self.capacity, seed=self.seed,
    #                      mem=self.mem[sliced])

    def save_to_pkl(self, filename):
        rng = self.rng
        self.rng = None
        backend = self.backend
        self.backend = None
        with open(filename, "wb") as out_file:
            pickle.dump(self, out_file)
        self.backend = backend
        self.rng = rng

    @staticmethod
    def load_from_pkl(filename):
        with open(filename, "rb") as in_file:
            return pickle.load(in_file)

    def append(self, s: Union[list, tuple, int, float],
               a: Union[list, tuple, int, float],
               r: float,
               s_: Union[list, tuple, int, float],
               d: float):
        if isinstance(s, (int, float)):  # state is an int
            s = [[s]]  # make state a list of sequences
            s_ = [[s_]]
        elif isinstance(s[0], (int, float)):  # state does not have multiple seq
            s = [s]
            s_ = [s_]
        for i in range(self.s_len):
            self.s[i][self.ptr] = self.backend.asarray(s[i], dtype=np.float32)
            self.s_[i][self.ptr] = self.backend.asarray(s_[i], dtype=np.float32)

        self.a[self.ptr] = self.backend.asarray(a, dtype=np.float32)
        self.r[self.ptr] = r
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class CupyMemory(Memory):

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

        super().__init__(capacity, state_shapes, action_shape, seed, backend=cp)

    def __str__(self):
        return f'module: {type(self).__name__} \n' \
               f'cupy device id: {cp.cuda.Device().id} \n' \
               f'capacity: {self.capacity} \n' \
               f'size: {self.size} \n' \
               f'current pointer: {self.ptr} \n' \
               f'seed: {self.seed} \n' \
               f'Shapes and types: \n' \
               f's: {[f"{i.shape} {type(i)} {i.device}" for i in self.s]} \n' \
               f'a: {self.a.shape} {type(self.a)} {self.a.device} \n' \
               f'r: {self.r.shape} {type(self.r)} {self.r.device} \n' \
               f's_: {[f"{i.shape} {type(i)} {i.device}" for i in self.s_]} \n' \
               f'd: {self.d.shape} {type(self.d)} {self.d.device} \n'

    @staticmethod
    def set_device(self, gpu_id=0):
        cp.cuda.Device(gpu_id).use()

    @staticmethod
    def get_device(self):
        return cp.cuda.Device().id

    @staticmethod
    def load_from_pkl(filename):
        with open(filename, "rb") as in_file:
            mem = pickle.load(in_file)
        mem.backend = cp
        mem.rng = mem.backend.random.default_rng(seed=mem.seed)
        return mem


class NumpyMemory(Memory):
    """
    Stores and returns list of numpy arrays for s a r s_ d

    assumes you are storing s a r s_ d : state action reward next_state done
    # state / next_state : tuple of n-dim numpy arrays
    # a : 1-d numpy array of floats or ints
    # r : 1-d numpy array of floats #TODO: rewards could also be n-dim
    # d : 1-d numpy array of booleans

    # TODO: if we made this tensor based instead of np, would it make it better ?
    """

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

        super().__init__(capacity, state_shapes, action_shape, seed)

        # from numpy.random import MT19937, RandomState, Generator
        # mt = MT19937(seed)
        # rs = RandomState(mt)
        # rg = Generator(mt)

        @staticmethod
        def load_from_pkl(filename):
            with open(filename, "rb") as in_file:
                mem = pickle.load(in_file)
            mem.backend = np
            mem.rng = mem.backend.random.default_rng(seed=mem.seed)
            return mem
