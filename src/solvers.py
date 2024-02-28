#!/usr/bin/env python3

"""
Solvers
"""

from enum import Enum

import numpy as np


class RootFindStatus(Enum):
  Success = 0
  Fail = 1


# End RootFindStatus


class Solvers:
  """
  Solver base class
  """

  def __init__(self, maxiters, fptol):
    self.maxiters = maxiters
    self.fptol = fptol

  # End __init__

  def check_bracket_(self, a, b, fa, fb):
    """
    check if root is in bracket
    """

    return RootFindStatus.Success if (fa * fb < 0.0) else RootFindStatus.Fail

  # End check_bracket_

  def fixed_point(self, func, x0, a, b):
    """
    Classical fixed point iteration
    """

    status = self.check_bracket_(a, b, func(a) - a, func(b) - b)
    if status == RootFindStatus.Fail:
      raise ValueError(
        f"No root in bracket! a, b, fa, fb = {a}, {b}, {func(a)}, {func(b)}"
      )
    # TODO: implement bisection and drop into bisection if fail

    n = 0
    error = 1.0
    ans = 0.0
    while n <= self.maxiters and error >= self.fptol:
      x1 = func(x0)
      error = abs(x1 - x0)
      x0 = x1
      n += 1

      if n == self.maxiters:
        print(" ! Not converged!")

      ans = x1
    return ans

  # End fixed_point

  def fixed_point_aa(self, func, x0, a, b):
    """
    Anderson accelerated fixed point iteration
    """

    # status = self.check_bracket_(a, b, func(a) - a, func(b) - b)
    status = RootFindStatus.Success
    if status == RootFindStatus.Fail:
      raise ValueError(
        f"No root in bracket! a, b, fa, fb = {a}, {b}, {func(a)}, {func(b)}"
      )
    # TODO: implement bisection and drop into bisection if fail

    # residual
    g = lambda x: func(x) - x

    n = 0
    error = 1.0
    xkm1 = 0.0
    xkp1 = 0.0
    xk = func(x0)
    xkm1 = x0
    while n <= self.maxiters and error >= self.fptol:
      alpha = -g(xk) / (g(xkm1) - g(xk))
      xkp1 = alpha * func(xkm1) + (1.0 - alpha) * func(xk)
      error = abs(xk - xkp1)
      xkm1 = xk
      xk = xkp1

      n += 1

      if n == self.maxiters:
        print(" ! Not converged!")
    return xk

  # End fixed_point
