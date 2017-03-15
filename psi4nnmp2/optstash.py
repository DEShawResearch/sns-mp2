from psi4 import core
from contextdecorator import ContextDecorator
from functools import wraps
import psi4.driver.p4util as p4util


class psiopts(ContextDecorator):
    def __init__(self, *psi_args):
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
            if len(k) == 1:
                core.set_global_option(k[0], v)
            else:
                core.set_local_option(k[0], k[1], v)

    def __exit__(self, *args, **kwargs):
        self.optstash.restore()
