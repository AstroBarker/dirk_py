#!/usr/bin/env python3

"""
Solvers
"""

from enum import Enum


class RootFindStatus(Enum):
  """
  Enum class for root find status
  Values:
    Success
    Fail
  """

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

    status = self.check_bracket_(a, b, func(a) - a, func(b) - b)
    if status == RootFindStatus.Fail:
      raise ValueError(
        f"No root in bracket! a, b, fa, fb = {a}, {b}, {func(a)}, {func(b)}"
      )
    # TODO: implement bisection and drop into bisection if fail

    # residual
    def g(x):
      return func(x) - x

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

  # End fixed_point_aa

  def newton(self, func, dfunc, x0, a, b):
    """
    Newton-Raphson iteration
    """
    status = self.check_bracket_(a, b, func(a) - a, func(b) - b)
    if status == RootFindStatus.Fail:
      raise ValueError(
        f"No root in bracket! a, b, fa, fb = {a}, {b}, {func(a)}, {func(b)}"
      )
    # TODO: implement bisection and drop into bisection if fail

    n = 0
    h = func(x0) / dfunc(x0)
    error = 1.0
    while n <= self.maxiters and error >= self.fptol:
      xn = x0
      h = func(xn) / dfunc(xn)
      x0 = xn - h
      error = abs(xn - x0)
      n += 1

      if n == self.maxiters:
        print(" ! Not converged!")
    return x0

  # End newton

  def newton_aa(self, func, dfunc, x0, a, b):
    """
    Anderson accelerated Newton-Raphson iteration
    """
    status = self.check_bracket_(a, b, func(a) - a, func(b) - b)
    if status == RootFindStatus.Fail:
      raise ValueError(
        f"No root in bracket! a, b, fa, fb = {a}, {b}, {func(a)}, {func(b)}"
      )
    # TODO: implement bisection and drop into bisection if fail

    n = 0
    h = func(x0) / dfunc(x0)
    error = 1.0
    xk = x0 - h
    xkm1 = x0
    xkp1 = 0.0
    while n <= self.maxiters and error >= self.fptol:
      hp1 = func(xk) / dfunc(xk)
      h = func(xkm1) / dfunc(xkm1)

      # Anderson acceleration step
      gamma = hp1 / (hp1 - h)

      xkp1 = xk - hp1 - gamma * (xk - xkm1 - hp1 + h)
      error = abs(xk - xkp1)

      xkm1 = xk
      xk = xkp1

      n += 1
      if n == self.maxiters:
        print(" ! Not converged!")
    return xk

  # End newton_aa
