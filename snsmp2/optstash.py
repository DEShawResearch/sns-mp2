# Copyright 2017  D. E. Shaw Research, LLC
# 
# All rights reserved.
# 
# D. E. Shaw Research, LLC ("DESRES") hereby grants you a limited,
# revocable license to use and/or modify the attached computer code (the
# "Code") for internal purposes only. Redistribution of any kind and in
# any form is strictly prohibited.
# 
# In consideration of the rights granted to you hereunder, you hereby
# agree to indemnify, defend and hold DESRES harmless from and against
# any and all damages sustained by DESRES, which damages arise out of or
# relate to your use of the Code, including without limitation any
# damages caused by any allegation or claim that the Code infringes the
# rights of any third party.
# 
# THE ATTACHED CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL DESRES BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS CODE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


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
