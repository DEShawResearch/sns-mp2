from __future__ import print_function


import os
import time
import itertools
import numpy as np
import psi4
from psi4 import core
import psi4.driver.p4util as p4util
from psi4.driver.procrouting.proc import run_dfmp2, run_dfmp2_gradient, scf_wavefunction_factory
from psi4.driver.procrouting.proc import scf_helper



# External Potential      Basis        Quality

# Monomer A               Monomer A    Q
# Monomer B               Monomer B    Q
# Monomer A               Dimer        T
# Monomer A               Dimer        Q
# Monomer B               Dimer        Q
# Monomer B               Dimer        Q
# Dimer                   Dimer        T    
# Dimer                   Dimer        Q


def run_nnmp2(name, molecule, **kwargs):
    core.tstart()
    # Force to c1
    molecule = molecule.clone()
    molecule.reset_point_group('c1')
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    nfrag = molecule.nfragments()
    if nfrag != 2:
        raise ValidationError('NN-MP2 requires active molecule to have 2 fragments, not %s.' % (nfrag))

    espx_fields = run_espx(name, molecule=molecule, basis='aug-cc-pvtz')
    core.print_out('\n')
    p4util.banner('Dimer Basis')
    core.print_out('\n')
    intene_fields = run_intene(name, molecule=molecule, lowbasis='aug-cc-pvtz', highbasis='aug-ccpvqz')

    outlines = [
        '',
        '-' * 77,
        '=' * 24 + ' DESRES ENERGY DECOMPOSITION ' + '=' * 24,
        '-' * 77,
    ]
    
    for k, v in itertools.chain(sorted(espx_fields.iteritems()), sorted(intene_fields.iteritems())):
        outlines.append('{:<52s} {:24.16f}'.format(k + ':', v))

    outlines.extend(['-' * 77, ''])
    core.print_out('\n'.join(outlines))
    core.tstop()


def run_intene(name, molecule, **kwargs):
    monomer1, monomer2 = dimerize(molecule, basis='dimer')
    m1low, m1high = run_dual_basis_mp2(name, monomer1)    
    m2low, m2high = run_dual_basis_mp2(name, monomer2)
    dlow, dhigh = run_dual_basis_mp2(name, molecule)

    def difference(field, basis):
        if basis == 'high':
            value = dhigh.get_variable(field) - (m1high.get_variable(field) + m2high.get_variable(field))
        elif basis == 'low':
            value = dlow.get_variable(field) - (m1low.get_variable(field) + m2low.get_variable(field))
        else:
            raise RuntimeError()
        return value

    lowbasis = os.path.basename(dlow.basisset().name()).replace('.gbs', '')
    highbasis = os.path.basename(dhigh.basisset().name()).replace('.gbs', '')

    value = {
        ('DF-MP2/%s CP Interaction Energy' % highbasis):               difference('Total Energy', basis='high'),
        ('DF-HF/%s CP Interaction Energy' % highbasis):                difference('Reference Energy', basis='high'),
        ('DF-MP2/%s CP Same-Spin Interaction Energy' % highbasis):     difference('Same-Spin Energy', basis='high'),
        ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % highbasis): difference('Opposite-Spin Energy', basis='high'),

        ('DF-MP2/%s CP Interaction Energy' % lowbasis):               difference('Total Energy', basis='low'),
        ('DF-HF/%s CP Interaction Energy' % lowbasis):                difference('Reference Energy', basis='low'),
        ('DF-MP2/%s CP Same-Spin Interaction Energy' % lowbasis):     difference('Same-Spin Energy', basis='low'),
        ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % lowbasis): difference('Opposite-Spin Energy', basis='low'),
    }
    return value


def run_espx(name, molecule, basis='aug-cc-pvqz', **kwargs):
    optstash = p4util.optproc.OptionsState(
        ['BASIS'],
    )
    core.set_global_option("BASIS", basis)


    monomer1, monomer2 = dimerize(molecule, basis='monomer')

    with timer('Monomer_A'):
        m1hf, m1mp2 = converge_monomer(basis=basis, molecule=monomer1)
    with timer('Monomer_B'):
        m2hf, m2mp2 = converge_monomer(basis=basis, molecule=monomer2)

    with timer('Decomposition_Setup'):
        decomp = EnergyDecomposition(monomer1, monomer2, molecule)

    with timer('Electrostatics_Overlap_Initialization'):
        esovlpfunc = decomp.esovlp()
    with timer('Electrostatics_Overal_Evaluation')
        eshf, ovlhf = esovlpfunc(m1hf, m2hf)
        esmp, ovlmp = esovlpfunc(m1mp2, m2mp2)

        # release some memory
        del esovlpfunc

    with timer('Heitler_London_Initialization'):
        hlfunc = decomp.hl()
    with timer('Heitler_London_Evaluation')
        hl = hlfunc(m1hf, m2hf)

    basis = os.path.basename(m1hf.basisset().name()).replace('.gbs', '')
    values = {
        'DF-HF/{} Electrostatic Interaction Energy'.format(basis): eshf,
        'DF-HF/{} Density Matrix Overlap'.format(basis): ovlhf,
        'DF-MP2/{} Electrostatic Interaction Energy'.format(basis): esmp,
        'DF-MP2{} Density Matrix Overlap'.format(basis): ovlmp,
        '{} Heitler-London Energy'.format(basis): hl
    }
    optstash.restore()
    return values


def converge_monomer(molecule, basis, ref_wfn=None):
    optstash = p4util.optproc.OptionsState(
        ['ONEPDM'],
    )
    core.set_global_option("ONEPDM", True)

    molecule.update_geometry()
    core.IO.set_default_namespace(molecule.name())
    core.print_out('\n')
    p4util.banner('%s Density Matrix' % molecule.name())
    core.print_out('\n')

    wfn_dfmp2 = run_dfmp2_gradient('df-mp2', molecule=molecule)
    ref_wfn = wfn_dfmp2.reference_wavefunction()

    # rmcgibbo
    #return scf_helper('hf', molecule=molecule), None

    optstash.restore()
    return ref_wfn, wfn_dfmp2


class HeitlerLondonFunctor(object):
    def __init__(self, energy_decomposition):
        self.p = energy_decomposition
        self.jk = self.p._initialize_jk(self.p.dimer_basis, self.p.dimer_aux_basis)

    def __del__(self):
        self.jk.finalize()

    def __call__(self, mol1_wfn, mol2_wfn):
        nbf = self.p.dimer_basis.nbf()
        nocc = mol1_wfn.nalpha() + mol2_wfn.nalpha()

        # Take the occupied orbitals from the two HF monomer wavefunctions
        # and pack them (block diagonal) into the dimer basis set.
        m1_OCC = mol1_wfn.Ca_subset('SO', 'OCC')
        m2_OCC = mol2_wfn.Ca_subset('SO', 'OCC')

        C = core.Matrix(nbf, nocc)
        C.zero()
        C.np[:mol1_wfn.nso(), :mol1_wfn.nalpha()] = m1_OCC.np[:, :]
        C.np[mol1_wfn.nso():, mol1_wfn.nalpha():] = m2_OCC.np[:, :]
        del m1_OCC, m2_OCC

        C = orthogonalize(C, self.p.dimer_S)
        # D2 = core.Matrix.doublet(C2, C2, False, True)

        # At this point, it should be the case that
        # C.T * S * C == I
        np.testing.assert_array_almost_equal(
            core.Matrix.triplet(C, self.p.dimer_S, C, True, False, False),
            np.eye(nocc))

        self.jk.C_clear()
        self.jk.C_left_add(C)

        with timer('hl jk'):
            self.jk.compute()

        J = self.jk.J()[0]
        K = self.jk.K()[0]
        D = self.jk.D()[0]

        # 2T + 2V + 2J - K

        FH = J.clone()
        FH.zero()
        FH.axpy(2, self.p.dimer_T)
        FH.axpy(2, self.p.dimer_V)
        FH.axpy(2, J)
        FH.axpy(-1, K)

        energy = FH.vector_dot(D) + self.p.dimer.nuclear_repulsion_energy()
        hl = energy - (mol1_wfn.energy() + mol2_wfn.energy())
        return hl


class EletrostaticsOverlapFunctor(object):
    def __init__(self, energy_decomposition):
        self.p = energy_decomposition
        self.jk = self.p._initialize_jk(self.p.dimer_basis, self.p.dimer_aux_basis, do_J=True, do_K=False)

    def __del__(self):
        self.jk.finalize()

    def __call__(self, mol1_wfn, mol2_wfn):
        nbf = self.p.dimer_basis.nbf()
        nbf1 = mol1_wfn.nso()
        nbf2 = mol2_wfn.nso()
        Ca = core.Matrix(nbf, max(nbf1, nbf2))
        Ca.zero()

        if mol1_wfn.name() == 'SCF':
            try:
                print(nbf1, mol1_wfn.nalpha(), mol1_wfn.Ca_subset('SO', 'OCC').shape)
                Ca.np[:nbf1, :mol1_wfn.nalpha()] = mol1_wfn.Ca_subset('SO', 'OCC')
            except:
                import IPython; IPython.embed()
        elif mol2_wfn.name() == 'DF-MP2':
            m1ca = mol1_wfn.Ca()
            Ca.np[:nbf1, :nbf1] = np.sqrt(np.maximum(0, mol1_wfn.epsilon_a())) * m1ca.np[:, :]
        else:
            raise ValueError('Unrecognized wfn: %s' % mol1_wfn.name())

        self.jk.C_clear()
        self.jk.C_left_add(Ca)

        with timer('esovlp jk 1'):
            self.jk.compute()

        J = self.jk.J()[0]
        D1 = self.jk.D()[0]

        J_1to2 = J.np[nbf1:, nbf1:]
        elel_1to2 = 2 * np.sum(J_1to2 * mol2_wfn.Da())
        nuel_1to2 = 2 * (self.p.dimer_V.vector_dot(D1) - self.p.monomer1_V.vector_dot(mol1_wfn.Da()))
        ovlp1 = core.Matrix.doublet(self.p.dimer_S, D1, False, False)

        Ca.zero()
        if mol2_wfn.name() == 'SCF':
            Ca.np[nbf1:, :mol2_wfn.nalpha()] = mol2_wfn.Ca_subset('SO', 'OCC')
        elif mol2_wfn.name() == 'DF-MP2':
            m2ca = mol2_wfn.Ca()
            Ca.np[nbf1:, :nbf2] = np.sqrt(np.maximum(0, mol2_wfn.epsilon_a())) * m2ca.np[:, :]
        else:
            raise ValueError('Unrecognized wfn: %s' % mol2_wfn.name())

        self.jk.C_clear()
        self.jk.C_left_add(Ca)

        with timer('esovlp jk 2'):
            self.jk.compute()

        J = self.jk.J()[0]
        D2 = self.jk.D()[0]

        J_2to1 = J.np[:nbf1, :nbf1]
        elel_2to1 = 2 * np.sum(J_2to1 * mol1_wfn.Da())
        nuel_2to1 = 2 * (self.p.dimer_V.vector_dot(D2) - self.p.monomer2_V.vector_dot(mol2_wfn.Da()))

        ovlp2 = core.Matrix.doublet(self.p.dimer_S, D2, False, False)

        overlap = 4 * np.sum(ovlp1.np * ovlp2.np.T)
        #assert abs(elel_1to2 - elel_2to1) < 1e-10
        # print('ELEL', elel_1to2, elel_2to1)

        nunu = self.p.dimer.nuclear_repulsion_energy() - (
            self.p.monomer1.nuclear_repulsion_energy() +
            self.p.monomer2.nuclear_repulsion_energy())

        electrostatic = nunu + nuel_1to2 + nuel_2to1 + elel_1to2 + elel_2to1
        return electrostatic, overlap


class EnergyDecomposition(object):
    def __init__(self, monomer1, monomer2, dimer):
        self.dimer = dimer
        self.monomer1 = monomer1
        self.monomer2 = monomer2

        namespace = core.IO.get_default_namespace()
        core.IO.set_default_namespace('Dimer')
        self.dimer_basis, self.dimer_aux_basis = self._initialize_basis(dimer)
        self.dimer_V, self.dimer_T, self.dimer_S = self._initialize_mints(self.dimer_basis)
        core.IO.set_default_namespace(namespace)

        monomer1_basis, _ = self._initialize_basis(self.monomer1)
        monomer2_basis, _ = self._initialize_basis(self.monomer2)
        self.monomer1_V = self._initialize_mints(monomer1_basis, v_only=True)
        self.monomer2_V = self._initialize_mints(monomer2_basis, v_only=True)

    def _initialize_basis(self, mol):
        basis = core.BasisSet.build(mol)
        aux_basis = core.BasisSet.build(mol, "DF_BASIS_MP2",
                                        core.get_option("DFMP2", "DF_BASIS_MP2"),
                                        "RIFIT", core.get_global_option("BASIS"))
        return basis, aux_basis

    def _initialize_jk(self, basis, aux_basis, do_J=True, do_K=True):
        jk = core.JK.build(basis, aux_basis)
        jk.set_memory(int(float(core.get_global_option("SCF_MEM_SAFETY_FACTOR")) * core.get_memory()) / 8)
        jk.set_do_J(do_J)
        jk.set_do_K(do_K)
        jk.print_header()
        jk.initialize()
        return jk

    def _initialize_mints(self, basis, v_only=False):
        mints = core.MintsHelper(basis)
        V = mints.ao_potential()
        if v_only:
            return V

        T = mints.ao_kinetic()
        S = mints.ao_overlap()
        return V, T, S

    def hl(self):
        return HeitlerLondonFunctor(self)

    def esovlp(self):
        return EletrostaticsOverlapFunctor(self)


def orthogonalize(C, S):
    nbf, nocc = C.shape

    eigenvectors = core.Matrix(nocc, nocc)
    eigvals = core.Vector(nocc)
    sqrt_eigvals = core.Vector(nocc)

    CTSC = core.Matrix.triplet(C, S, C, True, False, False)
    CTSC.diagonalize(eigenvectors, eigvals, core.DiagonalizeOrder.Ascending)

    orthonormal = core.Matrix.doublet(C, eigenvectors, False, False)

    sqrt_eigvals.np[:] = np.sqrt(eigvals.np)
    orthonormal.np[:, :] /= sqrt_eigvals.np[np.newaxis, :]

    return orthonormal


class timer(object):
    """A timing context manager
    Examples
    --------
    >>> long_function = lambda : None
    >>> with timing('long_function'):
    ...     long_function()
    long_function: 0.000 seconds
    """
    times = {}

    def __init__(self, name='block'):
        self.name = name
        self.time = 0
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, ty, val, tb):
        self.end = time.time()
        self.time = self.end - self.start
        print("%s: %0.3f seconds" % (self.name, self.time))

        #os.system('grep VmHWM /proc/%s/status' % os.getpid())
        #os.system('grep VmRSS /proc/%s/status' % os.getpid())

        self.times[self.name] = self.end - self.start
        return False


def run_dual_basis_mp2(name, molecule, lowbasis='aug-cc-pvtz', highbasis='aug-cc-pvqz', **kwargs):
    optstash = p4util.optproc.OptionsState(
        ['BASIS'],
        ['SCF', 'GUESS'],
    )
    guesspace = molecule.name() + '.low'
    core.IO.set_default_namespace(guesspace)

    # Run MP2 in the low basis set
    core.set_global_option('BASIS', lowbasis)
    low_wfn = run_dfmp2(name, molecule=molecule)

    # Move files to proper namespace
    namespace = molecule.name() + '.high'
    core.IO.change_file_namespace(180, guesspace, namespace)
    core.IO.set_default_namespace(namespace)

    # Run MP2 in the high basis set, using the low basis
    # result to acceperate convergence
    core.set_global_option('BASIS', highbasis)    
    core.set_local_option('SCF', 'GUESS', 'READ')
    high_wfn = run_dfmp2(name, molecule=molecule)

    optstash.restore()
    return low_wfn, high_wfn


def dimerize(molecule, basis='monomer'):
    if basis == 'monomer':
        monomer1 = molecule.extract_subsets(1)
        monomer1.set_name('Monomer_A_MCBS')
        monomer2 = molecule.extract_subsets(2)
        monomer2.set_name('Monomer_B_MCBS')
    elif basis == 'dimer':
        monomer1 = molecule.extract_subsets(1, 2)
        monomer1.set_name('Monomer_A_DCBS')
        monomer2 = molecule.extract_subsets(2, 1)
        monomer2.set_name('Monomer_B_DCBS')

    return monomer1, monomer2


# Integration with driver routines
psi4.driver.procedures['energy']['nnmp2'] = run_nnmp2
