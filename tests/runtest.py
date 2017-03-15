#!/usr/bin/env python
import os
import sys
import time
import pytest
import garden
import subprocess
import multiprocessing

THISDIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("inputfn", [
    "2x{CC}.in",
    "{CF}_O.in",
])
def test_generator(inputfn, cmd_prefix, pythonpath_append):
    fullname = os.path.join(THISDIR, inputfn)
    cmd = cmd_prefix + ['psi4', '-n', str(multiprocessing.cpu_count()), fullname, 'stdout']
    with open(os.path.join(THISDIR, 'testlog.out'), 'a', 0) as logfile:
        assert backtick(cmd, logfile, pythonpath_append=pythonpath_append) == 0


def backtick(exelist, loghandle, pythonpath_append=None):
    """Executes the command-argument list in *exelist*, directing the
    standard output to screen and file logfile and string p4out. Returns
    the system status of the call.
    """
    env = dict(os.environ)
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
    pytest.main(['--verbose', '--psi4nnmp2_version=local'])