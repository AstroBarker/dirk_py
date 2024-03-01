#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from dirk import DIRK

dt = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])


def func1(y):
  return -15.0 * y


def ans1(y):
  return np.exp(-15.0 * y)


def L2(data, theory):
  """
  L2 norm
  """
  return np.sum(np.power(data - theory, 2.0)) / len(data)


def convergence(mydirk):
  """
  Convergence for a given DIRK method
  """

  t_end = 2.0
  error = np.zeros(len(dt))

  for i in range(len(dt)):
    mydirk.evolve(func1, t_end, dt[i])
    theory = ans1(np.array(mydirk.time))
    error[i] = L2(mydirk.sol, theory)
  return error


# End convergence


def main():
  dirk11 = DIRK(1, 1)
  error11 = convergence(dirk11)

  dirk22 = DIRK(2, 2)
  error22 = convergence(dirk22)

  dirk33 = DIRK(3, 3)
  error33 = convergence(dirk33)

  dirk43 = DIRK(4, 3)
  error43 = convergence(dirk43)

  fig, ax = plt.subplots()

  ax.loglog(dt, error11, color="teal", label="DIRK(1,1)", ls=" ", marker="x")

  ax.loglog(dt, error22, color="#400080", label="DIRK(2,2)", ls=" ", marker="o")
  ax.loglog(dt, error33, color="#800000", label="DIRK(3,3)", ls=" ", marker="o")
  ax.loglog(dt, error43, color="#408000", label="DIRK(4,3)", ls=" ", marker="o")

  ax.legend(frameon=True)
  ax.set(xlabel="dt", ylabel=r"$L_{2}$")

  plt.savefig("convergence.png")


# End main


if __name__ == "__main__":
  main()
