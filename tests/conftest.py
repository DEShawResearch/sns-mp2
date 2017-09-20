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
import os
import pytest
THISDIR = os.path.abspath(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption("--psi4nnmp2_version", default='',
                     help=("Version of psi4nnmp2 to test "
                           "Available values: 'local', or "
                           "a garden tag"))

def pytest_generate_tests(metafunc):
    psi4nnmp2_version = metafunc.config.option.psi4nnmp2_version
    if not psi4nnmp2_version:
        psi4nnmp2_version = 'local'
    metafunc.parametrize("psi4nnmp2_version", [psi4nnmp2_version],
                         scope='session')


@pytest.yield_fixture(scope='session')
def pythonpath_append(psi4nnmp2_version):
    env = os.environ.copy()
    if psi4nnmp2_version == 'local':
        return os.path.join(THISDIR, '..')
    return None

@pytest.yield_fixture(scope='session')
def cmd_prefix(psi4nnmp2_version):
    if psi4nnmp2_version == 'local':
        prereqs = (s.strip() for s in open(os.path.join(
            THISDIR, '..', 'garden-prereq.txt')).readlines())
        return('garden with -m ' + ' -m '.join(prereqs)).split()

    return ['garden', 'with', '-c', '-m', psi4nnmp2_version]
