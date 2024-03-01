# dirk_py (diagonally implicit Runge Kutta in python)
Or, Dang, I Really Know PYthon?

Python 3.x test implementation of diagonally implicit runge kutta integrator.
Based on Butcher form.
I like [Gottlieb et al. 2009](https://link.springer.com/article/10.1007/s10915-008-9239-z) and [Conde et al. 2017](https://arxiv.org/abs/1702.04621).


# Code Style
Code linting and formatting is done with [ruff](https://docs.astral.sh/ruff/).
Rules are listed in [ruff.toml](ruff.toml).
To check all python in the current directory, you may `ruff .`.
To format a given file according to `ruff.toml`, run `ruff format file.py`.
Checks for formatting are performed on each push / PR.
