# from abc import ABC, abstractmethod
# import numpy as np

# # NOTE: not tested


# class EnsembleFilter(ABC):
#     def __init__(self, M, Q, h, R):
#         self.M = M
#         self.Q = Q
#         self.h = h
#         self.R = R

#         self.Nx = Q.shape[0]
#         self.Ny = R.shape[0]

#         # TODO:

#     def initialize(self, X0):
#         Nx, m = X0.shape
#         assert Nx == self.Nx

#         self.m = m
#         self.t = 0.0
#         self.X = X0

#     @abstractmethod
#     def forecast(self):
#         pass

#     @abstractmethod
#     def update(self, y_obs):
#         pass
