from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

# NOTE: not tested


class EnsembleFilter(ABC):
    def __init__(self, Dynamics, Observation):
        self.Dynamics = Dynamics
        self.Observation = Observation

        # TODO:

    @abstractmethod
    def initialize(self, X0: NDArray):
        dim_x, m = X0.shape
        self.dim_x = dim_x
        self.m = m
        self.t = 0.0
        self.X = X0

    @abstractmethod
    def forecast(self):
        # TODO: 各memberを発展
        pass

    @abstractmethod
    def update(self, y_obs: NDArray):
        pass
