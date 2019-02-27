#                       SNS-MP2 LICENSE AGREEMENT
# 
# Copyright 2017, D. E. Shaw Research. All rights reserved.
# 
# Redistribution and use of (1) the SNS-MP2 software in source and binary forms
# and (2) the associated electronic structure data released with the software,
# with or without modification, is permitted provided that the following
# conditions are met:
# 
#     * Redistributions of source code and the associated data must retain the
#     above copyright notice, this list of conditions, and the following
#     disclaimer.
# 
#     * Redistributions in binary form must reproduce the above copyright 
#     notice, this list of conditions, and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
# 
# Neither the name of D. E. Shaw Research nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE AND DATA ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDINGNEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE AND/OR DATA, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__all__ = ['__version__']

import psi4
import psi4.driver

from pkg_resources import parse_version
if parse_version(psi4.__version__) < parse_version('1.3rc2'):
    raise ImportError('Psi4 {:s} is not compatible (v1.3+ necessary)'.format(psi4.__version__))

from .snsmp2 import run_sns_mp2
psi4.driver.procedures['energy']['sns-mp2'] = run_sns_mp2


def _get_version():
    import pkg_resources
    resource = pkg_resources.Requirement.parse('snsmp2')
    provider = pkg_resources.get_provider(resource)
    return provider.version


try:
    __version__ = _get_version()
except ImportError:
    __version__ = ''
    pass

del _get_version
