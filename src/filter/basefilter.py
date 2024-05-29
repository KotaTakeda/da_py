"""
Base class of filters
Assume the following state space model:
dx_t = f(x_t)dt + dv_t
y_t = h(x_t)+ w_t

where q q^t = Q, r r^t = R
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

# NOTE: not tested


class BaseFilter(ABC):
    def __init__(self):
        # TODO:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def forecast(self):
        pass

    @abstractmethod
    def update(self, y_obs: NDArray):
        pass
