import psi4
import psi4.driver
from .snsmp2 import run_sns_mp2
psi4.driver.procedures['energy']['sns-mp2'] = run_sns_mp2
