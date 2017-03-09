import psi4
import psi4.driver
from .nnmp2 import run_nnmp2
psi4.driver.procedures['energy']['nnmp2'] = run_nnmp2
