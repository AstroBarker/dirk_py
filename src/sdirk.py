"""
Class for IRK
"""


class IRK:
  """
  Class for IRK solver
  """

  def __init__(self, nStages, tOrder):
    assert nStages >= tOrder, "nStages >= tOrder"

    self.nStages = nStages
    self.tOrder = tOrder

    # setup tableau

    self.a_ij = np.zeros(nStages, nStages)
    self.b_i = np.zeros(nStages)
    self.c_i = np.zeros(nStages)


    if nStages == 1 and tOrder == 1:
      a[0, 0] = 1.0
      b[0] = 1.0
      c[0] = 1.0

  # End __init__

  def __str__(self):
    print(f"Implicit RK method: ")
    print(f"nStages : {self.nStages}")
    print(f"tOrder  : {self.tOrder}")
    return ""
  # End __str__

  def compute_increment(self, f):
    """
    Given IRk tableau and function f of du/dt = f(u), compute u^(n+1)
    """
  
# End IRK
