import os
import sys
import time
import pytest
import subprocess
import multiprocessing
import garden
garden.load('psi4nnmp2/0.1.0/bin')
THISDIR = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.parametrize("inputfn", [
    "2x{CC}.in",
    "{CF}_O.in",
])
def test_generator(inputfn):
    with open(os.path.join(THISDIR, 'testlog.out'), 'a', 0) as logfile:
        fullname = os.path.join(THISDIR, inputfn)
        assert backtick(['psi4', '-n', str(multiprocessing.cpu_count()), fullname],
                        logfile) == 0



def backtick(exelist, loghandle):
    """Executes the command-argument list in *exelist*, directing the
    standard output to screen and file logfile and string p4out. Returns
    the system status of the call.
    """
    try:
        retcode = subprocess.Popen(exelist, bufsize=0, stdout=subprocess.PIPE, universal_newlines=True)
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

