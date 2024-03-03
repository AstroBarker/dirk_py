#!/usr/bin/env python3

import numpy as np


def shuosher_to_butcher(alpha, beta):
  """
  Convert the Shu-Osher alpha - beta form to Butcher tabelau A, b
  """
  s = len(alpha[0, :])
  X = np.identity(s) - alpha[:-1, :]
  invX = np.linalg.inv(X)
  A = np.matmul(invX, beta[:-1, :])

  b = beta[-1, :] + np.matmul(alpha[-1, :], A)
  c = np.zeros(len(b))  # unused, but constructed
  for i in range(len(c)):
    c[i] = np.sum(A[i, :])

  return A, b, c


# End shuosher_to_butcher


def butcher_to_shuosher(A, b, c):
  """
  Convert Butcher tableau A, b to Shu-Osher form alpha, beta
  """
  s = len(b)
  r = radius_absolute_monotonicity(A, b, c)
  K = np.concatenate((A, [b]), axis=0)
  K = np.c_[K, np.zeros(s + 1)]
  e = np.ones(s + 1)

  beta = np.matmul(K, np.linalg.inv(np.identity(s + 1) + r * K))
  alpha = r * beta
  v = np.matmul((np.identity(s + 1) - alpha), e)
  return v, alpha, beta


# End butcher_to_shuosher


def radius_absolute_monotonicity(A, b, c):
  """
  Compute the radius of absolute monotonicity for given Butcher tableau
  See: https://github.com/ketch/RK-Opt/blob/master/RKtools/am_radius.m
  """
  eps = 1.0e-14
  rmax = 5000.0
  s = len(b)
  e_m = np.ones(s)

  K = np.concatenate((A, [b]), axis=0)
  rlo = 0.0
  rhi = rmax

  while abs(rhi - rlo) > eps:  # bisection solve
    r = 0.5 * (rhi + rlo)
    X = np.identity(s) + r * A
    beta = np.matmul(K, np.linalg.inv(X))
    ech = r * np.matmul(K, np.matmul(np.linalg.inv(X), e_m))
    if beta.min() < -1.0e-14 or ech.max() > 1.0 + 1e-14:
      rhi = r
    else:
      rlo = r
  return rlo


# End radius_absolute_monotonicity


def main():
  x = 0.0
  alpha = np.zeros((3, 2))
  alpha[1, 0] = 1.0
  alpha[2, 1] = 1.0
  beta = np.zeros((3, 2))
  beta[0, 0] = 1.0 / 4.0
  beta[1, 0] = 1.0 / 4.0
  beta[1, 1] = 1.0 / 4.0
  beta[2, 1] = 1.0 / 4.0

  A, b, c = shuosher_to_butcher(alpha, beta)
  r = radius_absolute_monotonicity(A, b, c)
  (
    v,
    alpha2,
    beta2,
  ) = butcher_to_shuosher(A, b, c)
  print(alpha2)
  print(beta)


# end main


if __name__ == "__main__":
  main()
