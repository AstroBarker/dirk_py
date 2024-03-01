#!/usr/bin/env python3

"""
Class for DIRK
"""

import sys

import numpy as np
from solvers import Solvers


def func1(y):
  return -15.0 * y


def dfunc1():
  return -15.0


def ans1(y):
  return np.exp(-15.0 * y)


class DIRK:
  """
  Class for DIRK solver
  """

  def __init__(self, nStages, tOrder, dt):
    assert dt > 0.0, "dt must be > 0.0"

    self.nStages = nStages
    self.tOrder = tOrder
    self.dt = dt

    # setup tableau

    self.a_ij = np.zeros((nStages, nStages))
    self.b_i = np.zeros(nStages)
    self.c_i = np.zeros(nStages)

    # Backward Euler tableau
    if nStages == 1 and tOrder == 1:
      self.a_ij[0, 0] = 1.0
      self.b_i[0] = 1.0
      self.c_i[0] = 1.0

    # L-stable
    if nStages == 2 and tOrder == 2:
      x = 1.0 - np.sqrt(2.0) / 2.0
      self.a_ij[0, 0] = x
      self.a_ij[1, 0] = 1.0 - x
      self.a_ij[1, 1] = x
      self.b_i[0] = 1.0 - x
      self.b_i[1] = x

    # L-stable
    if nStages == 3 and tOrder == 3:
      x = 0.4358665215
      self.a_ij[0, 0] = x
      self.a_ij[1, 0] = (1.0 - x) / 2.0
      self.a_ij[2, 0] = (-3.0 * x * x / 2.0) + 4.0 * x - 0.25
      self.a_ij[1, 1] = x
      self.a_ij[2, 1] = (3.0 * x * x / 2.0) - 5.0 * x + 5.0 / 4.0
      self.a_ij[2, 2] = x
      self.b_i[0] = (-3.0 * x * x / 2.0) + 4.0 * x - 0.25
      self.b_i[1] = (3.0 * x * x / 2.0) - 5.0 * x + 5.0 / 4.0
      self.b_i[2] = x

    # L-stable
    if nStages == 4 and tOrder == 3:
      self.a_ij[0, 0] = 0.5
      self.a_ij[1, 0] = 1.0 / 6.0
      self.a_ij[2, 0] = -0.5
      self.a_ij[3, 0] = 1.5
      self.a_ij[1, 1] = 0.5
      self.a_ij[2, 1] = 0.5
      self.a_ij[3, 1] = -1.5
      self.a_ij[2, 2] = 0.5
      self.a_ij[3, 2] = 0.5
      self.a_ij[3, 3] = 0.5
      self.b_i[0] = 1.5
      self.b_i[1] = -1.5
      self.b_i[2] = 0.5
      self.b_i[3] = 0.5

    # storage
    self.U = 1.0
    self.U_s = np.zeros(nStages)  # stage storage

    # solver
    max_iters = 100
    fptol = 1.0e-15
    self.solver = Solvers(max_iters, fptol)

  # End __init__

  def __str__(self):
    print(f"Implicit RK method: ")
    print(f"nStages : {self.nStages}")
    print(f"tOrder  : {self.tOrder}")
    return ""

  # End __str__

  def update_(self, f):
    """
    Given DIRK tableau and rhs function f of du/dt = f(u), compute u^(n+1)
    """

    for i in range(self.nStages):
      # Solve u^(i) = dt a_ii f(u^(i))
      target = lambda u: self.dt * self.a_ij[i, i] * f(u)
      self.U_s[i] = self.solver.fixed_point_aa(target, self.U, -10.0, 10.0)

      # increment u^(i) by other bits
      for j in range(i):
        # print(i, " ", j)
        self.U_s[i] += self.dt * self.a_ij[i, j] * f(self.U_s[j])
      self.U_s[i] += self.U

    # u^(n+1) from the stages
    for i in range(self.nStages):
      self.U += self.dt * self.b_i[i] * f(self.U_s[i])

  # End compute_increment_

  def evolve(self, f, t_end):
    """
    timestepper
    """
    t = 0.0
    step = 0
    while t < t_end:
      if t + self.dt > t_end:
        self.dt = t_end - t

      self.update_(f)
      step += 1

      t += self.dt

  # End evolve


# End DIRK
if __name__ == "__main__":
  # main

  dirk = DIRK(1, 1, 0.1)
  dirk.evolve(func1, 1.0)
  print(dirk.U)
  print(ans1(1.0))
# End __main__
