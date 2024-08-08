#pip install "pybamm[plot,cite]" -q    # install PyBaMM if it is not installed
import pybamm
import matplotlib.pyplot as plt

model_dfn = pybamm.lithium_ion.DFN()
sim_dfn = pybamm.Simulation(model_dfn)
sim_dfn.solve([0, 3600])