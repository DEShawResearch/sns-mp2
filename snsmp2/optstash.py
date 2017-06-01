from psi4 import core
import psi4.driver.p4util as p4util

import re
from .contextdecorator import ContextDecorator
from functools import wraps


class psiopts(ContextDecorator):
    """This class can be used as a context manager or decorator to set options for the Psi4 core
    and restore the prior option state on exit.

    Example
    -------
    with psiopts('SCF_TYPE DF', 'MP2_TYPE DF', 'ONEPDM TRUE'):
        ref_wfn = scf_helper('scf', molecule=molecule)
        wfn = run_dfmp2('df-mp2', molecule=molecule, ref_wfn=ref_wfn)
    """
    def __init__(self, *psi_args):
        """
        Arguments
        ---------
        psi_args : list of strings
            Each psi_arg should contain two or three space separated tokens.
            2 tokens for setting a global option (e.g. 'ONEPDM TRUE') and
            3 tokens for setting a module optin (e.g. 'SCF GUESS READ').
        """

        psikwargs = dict()

        for arg in psi_args:
            split = arg.split()
            if len(split) not in {2,3}:
                raise ValueError('Malformed arg: "%s"' % arg)
            key = tuple(e.upper() for e in split[:-1])
            val = split[-1]
            psikwargs[key] = val

        self.psikwargs = psikwargs
        self.optstash = None
        
    def __enter__(self):
        self.optstash = p4util.optproc.OptionsState(*self.psikwargs.keys())
        for k, v in self.psikwargs.items():

            # Integer options need to be passed as an integer type,
            # or else it fails to set and doesn't throw an error.
            if re.match('^\d+$', v) is not None:
                v = int(v)

            if len(k) == 1:
                core.set_global_option(k[0], v)
            else:
                core.set_local_option(k[0], k[1], v)

    def __exit__(self, *args, **kwargs):
        self.optstash.restore()
