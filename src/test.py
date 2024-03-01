#!/usr/bin/env python3

import numpy as np

from solvers import Solvers
from dirk import DIRK, func1, ans1


def func_fp(x):
  return np.cos(x)


def func_newton(x):
  return np.cos(x) - x


def dfunc_newton(x):
  return -np.sin(x) - 1.0


def test_fp():
  """
  Test fixed point iteration
  """

  max_iters = 100
  fptol = 1.0e-8

  solver = Solvers(max_iters, fptol)

  x0 = 1.0
  a = 0.0
  b = 1.0

  my_ans = solver.fixed_point(func_fp, x0, a, b)

  ANS = 0.739085133215160641

  assert abs(my_ans - ANS) < fptol, f"Root must be within {fptol}"


# End test_fp


def test_fp_aa():
  """
  Test Anderson accelerated fixed point iteration
  """

  max_iters = 100
  fptol = 1.0e-8

  solver = Solvers(max_iters, fptol)

  x0 = 1.0
  a = 0.0
  b = 1.0

  my_ans = solver.fixed_point_aa(func_fp, x0, a, b)

  ANS = 0.739085133215160641

  assert abs(my_ans - ANS) < fptol, f"Root must be within {fptol}"


# End test_fp_aa


def test_newton():
  """
  Test newton iteration
  """

  max_iters = 100
  fptol = 1.0e-8

  solver = Solvers(max_iters, fptol)

  x0 = 1.0
  a = 0.0
  b = 1.0

  my_ans = solver.newton(func_newton, dfunc_newton, x0, a, b)

  ANS = 0.739085133215160641

  assert abs(my_ans - ANS) < fptol, f"Root must be within {fptol}"


def test_newton_aa():
  """
  Test newton iteration
  """

  max_iters = 100
  fptol = 1.0e-8

  solver = Solvers(max_iters, fptol)

  x0 = 1.0
  a = 0.0
  b = 1.0

  my_ans = solver.newton_aa(func_newton, dfunc_newton, x0, a, b)

  ANS = 0.739085133215160641

  assert abs(my_ans - ANS) < fptol, f"Root must be within {fptol}"


def test_dirk22():
  dirk = DIRK(2, 2, 0.1)
  dirk.evolve(func1, 1.0)

  answer = ans1(1.0)
  TOL = 1.0e-4

  assert abs(dirk.U - answer) <= TOL, f"Answer must be within {TOL}"


if __name__ == "__main__":
  # main
  test_fp()
  test_fp_aa()
  test_dirk22()
  test_newton()
  test_newton_aa()
