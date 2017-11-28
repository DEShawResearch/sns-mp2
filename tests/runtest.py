#!/usr/bin/env python
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
import sys
import time
import pytest
import subprocess
import multiprocessing
import shlex

THISDIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("inputfn", [
    "2x{CC}.in",
    "{CF}_O.in",
])
def test_generator(inputfn, cmd_prefix, pythonpath_append):
    fullname = os.path.join(THISDIR, inputfn)
    cmd = shlex.split(cmd_prefix) + ['psi4', '-n', str(multiprocessing.cpu_count()), fullname, 'stdout']
    with open(os.path.join(THISDIR, 'testlog.out'), 'a', 0) as logfile:
        assert backtick(cmd, logfile, pythonpath_append=pythonpath_append) == 0


def backtick(exelist, loghandle, pythonpath_append=None):
    """Executes the command-argument list in *exelist*, directing the
    standard output to screen and file logfile and string p4out. Returns
    the system status of the call.
    """
    env = dict(os.environ)
    env['TEST_SNSMP2'] = '1'
    if pythonpath_append:
        env['PYTHONPATH'] = '%s:%s' % (os.environ['PYTHONPATH'], pythonpath_append)

    try:
        retcode = subprocess.Popen(exelist, bufsize=0, stdout=subprocess.PIPE,
                                   universal_newlines=True,
                                   env=env)
    except OSError as e:
        sys.stderr.write('Command %s execution failed: %s\n' % (exelist, e.strerror))
        sys.exit(1)

    p4out = ''
    while True:
        data = retcode.stdout.readline()
        if not data:
            break
        sys.stdout.write(data)  # screen
        loghandle.write(data)  # file
        loghandle.flush()
        p4out += data  # string
    while True:
        retcode.poll()
        exstat = retcode.returncode
        if exstat is not None:
            return exstat
        time.sleep(0.1)
    loghandle.close()


def setup_module(module):
    os.chdir(THISDIR)

    try:
        os.unlink(os.path.join(THISDIR, 'testlog.out'))
    except OSError:
        pass


def teardown_module(module):
    try:
        os.unlink(os.path.join(THISDIR, 'timer.dat'))
    except OSError:
        pass


if __name__ == '__main__':
    pytest.main()
