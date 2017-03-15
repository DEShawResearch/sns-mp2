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
