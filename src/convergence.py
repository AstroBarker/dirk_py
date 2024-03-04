#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from dirk import DIRK

dt = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
dt = np.array([1e-4, 1e-3, 1e-2, 1e-1])


def func1(y):
  return 2.0 * y


def ans1(y):
  return np.exp(2.0 * y)


def L2(data, theory):
  """
  L2 norm
  """
  return np.sum(np.power(data - theory, 2.0)) / len(data)


def convergence(mydirk):
  """
  Convergence for a given DIRK method
  """

  t_end = 1.0
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

  fig, ax = plt.subplots()

  slope, intercept, r, p, se = stats.linregress(np.log10(dt), np.log10(error11))
  ax.plot(
    np.log10(dt),
    np.log10(error11),
    color="teal",
    label="DIRK(1,1)",
    ls=" ",
    marker="x",
  )
  # ax.loglog(dt, slope*dt+intercept, color="teal")
  # ax.loglog(dt, 1.0 * error11, color="teal")
  ax.plot(
    np.log10(dt),
    slope * np.log10(dt) + intercept,
    color="teal",
    label=f"DIRK(1,1): {slope:.3f}",
  )

  slope, intercept, r, p, se = stats.linregress(np.log10(dt), np.log10(error22))
  ax.plot(
    np.log10(dt),
    np.log10(error22),
    color="#400080",
    label="DIRK(2,2)",
    ls=" ",
    marker="x",
  )
  ax.plot(
    np.log10(dt),
    slope * np.log10(dt) + intercept,
    color="#400080",
    label=f"DIRK(2,2): {slope:.3f}",
  )

  slope, intercept, r, p, se = stats.linregress(np.log10(dt), np.log10(error33))
  ax.plot(
    np.log10(dt),
    np.log10(error33),
    color="#800000",
    label="DIRK(3,3)",
    ls=" ",
    marker="x",
  )
  ax.plot(
    np.log10(dt),
    slope * np.log10(dt) + intercept,
    color="#800000",
    label=f"DIRK(3,3): {slope:.3f}",
  )

  # slope, intercept, r, p, se = stats.linregress(np.log10(dt), np.log10(error43))
  # ax.plot(np.log10(dt), np.log10(error43), color="#408000", label="DIRK(4,3)", ls=" ", marker="x")
  # ax.plot(np.log10(dt), slope * np.log10(dt) + intercept, color="#408000", label=f"DIRK(4,3): {slope:.3f}")

  ax.legend(frameon=True)
  ax.set(xlabel=r"log$_{10}(dt)$", ylabel=r"log$_{10}$(L$_{2})$")

  plt.savefig("convergence.png")


# End main

if __name__ == "__main__":
  main()
