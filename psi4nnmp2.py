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
from psi4.driver.molutil import constants
from psi4 import extras

# External Potential      Basis        Quality

# Monomer A               Monomer A    Q
# Monomer B               Monomer B    Q
# Monomer A               Dimer        T
# Monomer A               Dimer        Q
# Monomer B               Dimer        T
# Monomer B               Dimer        Q
# Dimer                   Dimer        T
# Dimer                   Dimer        Q



#
# MA_DIMER_Q  MB_DIMER_Q
#    ^             ^
#    |             |
# MA_DIMER_T   MB_DIMER_T
#    ^             ^
#    |             |
# MA_MA_Q      MB_MB_Q


from collections import namedtuple
calcid = namedtuple('calcid', ('V', 'B', 'Z'))

class WavefunctionCache(object):
    def __init__(self, dimer, no_reuse=False, low='cc-pvtz', high='aug-cc-pvtz'):
        self._d = dimer
        self._m1d, self._m2d = dimerize(dimer, basis='dimer')
        self._m1m, self._m2m = dimerize(dimer, basis='monomer')
        self.no_reuse = no_reuse
        self.wfn_cache = {}
        self.basis_sets = {
            'low': low,
            'high': high,
        }

    def molecule(self, calc):
        if calc.V == 'm1' and calc.B == 'm':
            return self._m1m
        if calc.V == 'm2' and calc.B == 'm':
            return self._m2m
        if calc.V == 'm1' and calc.B == 'd':
            return self._m1d
        if calc.V == 'm2' and calc.B == 'd':
            return self._m2d
        if calc.V == 'd':
            return self._d
        raise ValueError(calc)

    def fmt_ns(self, calc):
        return '%s-%s-%s' % (calc.V, calc.B, calc.Z)

    def _init_ns(self, calc):
        # Move into new namespace
        core.IO.set_default_namespace(self.fmt_ns(calc))
        if self.no_reuse:
            return

        if calc.Z == 'high':
            # If high basis, look for low basis equivalent
            candidate = calcid(calc.V, calc.B, 'low')
            if candidate in self.wfn_cache:
                return self._init_upcast_C(oldcalc=candidate, calc=calc)

        if calc.B == 'd':
            # If in dimer basis, look for a similar calculation in the monomer basis
            candidates = [calcid(calc.V, 'm', 'low'), calcid(calc.V, 'm', 'high')]
            if calc.Z == 'high':
                # if we're currently in the high basis, reverse the candidates
                # to prefer a candidate in the highbasis
                candidates = reversed(candidates)

            for c in candidates:
                if c in self.wfn_cache:
                    return self._init_addghost_C(oldcalc=c, calc=calc)

        if calc.V == 'd' and calc.B == 'd':
            # Dimer in the dimer basis set
            # Look for monomers
            candidate1 = calcid('m1', 'd', calc.Z)
            candidate2 = calcid('m2', 'd', calc.Z)
            if candidate1 in self.wfn_cache and candidate2 in self.wfn_cache:
                return self._init_stack_C(calc, candidate1, candidate2)
                # pass

        print('No guess found', calc)
        # return None, None

    def _init_upcast_C(self, oldcalc, calc):
        print('Upcasting', oldcalc, calc)
        assert oldcalc.V == calc.V and oldcalc.B == calc.B
        core.set_local_option('SCF', 'GUESS', 'READ')
        core.IO.change_file_namespace(constants.PSIF_SCF_MOS, self.fmt_ns(oldcalc), self.fmt_ns(calc))

    def _init_addghost_C(self, oldcalc, calc):
        print('Adding ghost', oldcalc, calc)
        this_molecule = self.molecule(calc)
        old_molecule = self.molecule(oldcalc)

        old_filename = "%s.%s.npz" % (core.get_writer_file_prefix(old_molecule.name()), constants.PSIF_SCF_MOS)
        data = np.load(old_filename)
        Ca_occ = core.Matrix.np_read(data, "Ca_occ")
        Cb_occ = core.Matrix.np_read(data, "Cb_occ")

        m1_nso = self.wfn_cache[('m1', 'm', oldcalc.Z)].nso()
        m2_nso = self.wfn_cache[('m2', 'm', oldcalc.Z)].nso()
        m1_nalpha = self.wfn_cache[('m1', 'm', oldcalc.Z)].nalpha()
        m2_nalpha = self.wfn_cache[('m2', 'm', oldcalc.Z)].nalpha()
        m1_nbeta = self.wfn_cache[('m1', 'm', oldcalc.Z)].nbeta()
        m2_nbeta = self.wfn_cache[('m2', 'm', oldcalc.Z)].nbeta()

        if calc.V == 'm1':
            Ca_occ_d = core.Matrix('Ca_occ', (m1_nso + m2_nso), m1_nalpha)
            Ca_occ_d.np[:m1_nso, :] = Ca_occ.np[:, :]

            Cb_occ_d = core.Matrix('Cb_occ', (m1_nso + m2_nso), m1_nbeta)
            Cb_occ_d.np[:m1_nso, :] = Cb_occ.np[:, :]
        elif calc.V == 'm2':
            Ca_occ_d = core.Matrix('Ca_occ', (m1_nso + m2_nso), m2_nalpha)
            Ca_occ_d.np[-m2_nso:, :] = Ca_occ.np[:, :]

            Cb_occ_d = core.Matrix('Cb_occ', (m1_nso + m2_nso), m2_nbeta)
            Cb_occ_d.np[-m2_nso:, :] = Cb_occ.np[:, :]

        data_dict = dict(data)
        data_dict.update(Ca_occ_d.np_write(prefix='Ca_occ'))
        data_dict.update(Cb_occ_d.np_write(prefix='Cb_occ'))

        write_filename = core.get_writer_file_prefix(this_molecule.name()) + ".180.npz"
        np.savez(write_filename, **data_dict)
        extras.register_numpy_file(write_filename)
        core.set_local_option('SCF', 'GUESS', 'READ')


    def _init_stack_C(self, calc, oldcalc_m1, oldcalc_m2):
        assert oldcalc_m1.V == 'm1'
        assert oldcalc_m2.V == 'm2'
        print('Stacking monomer wfns', calc, oldcalc_m1, oldcalc_m2)
        this_molecule = self.molecule(calc)
        mol1 = self.molecule(oldcalc_m1)
        mol2 = self.molecule(oldcalc_m2)

        m1_C_fn = "%s.%s.npz" % (core.get_writer_file_prefix(mol1.name()), constants.PSIF_SCF_MOS)
        m2_C_fn = "%s.%s.npz" % (core.get_writer_file_prefix(mol2.name()), constants.PSIF_SCF_MOS)        
        m1_data = np.load(m1_C_fn)
        m2_data = np.load(m2_C_fn)
        m1_Ca_occ = core.Matrix.np_read(m1_data, "Ca_occ")
        m1_Cb_occ = core.Matrix.np_read(m1_data, "Cb_occ")
        m2_Ca_occ = core.Matrix.np_read(m2_data, "Ca_occ")
        m2_Cb_occ = core.Matrix.np_read(m2_data, "Cb_occ")

        m1_nso, m1_nalpha = m1_Ca_occ.shape
        m2_nso, m2_nalpha = m2_Ca_occ.shape
        m1_nbeta = m1_Cb_occ.shape[1]
        m2_nbeta = m2_Cb_occ.shape[1]
        assert m1_nso == m2_nso

        d_Ca_occ = core.Matrix('Ca_occ', (m1_nso), (m1_nalpha + m2_nalpha))
        d_Cb_occ = core.Matrix('Cb_occ', (m1_nso), (m1_nbeta + m2_nbeta))

        d_Ca_occ.np[:, :m1_nalpha] = m1_Ca_occ.np[:, :]
        d_Ca_occ.np[:, -m2_nalpha:] = m2_Ca_occ.np[:, :]

        d_Cb_occ.np[:, :m1_nbeta] = m1_Cb_occ.np[:, :]
        d_Cb_occ.np[:, -m2_nbeta:] = m2_Cb_occ.np[:, :]

        assert m1_data['symmetry'] == m2_data['symmetry'] == 'c1'
        assert m1_data['reference'] == m2_data['reference']
        assert m1_data['BasisSet'] == m2_data['BasisSet']
        assert m1_data['BasisSet PUREAM'] == m2_data['BasisSet PUREAM']

        data = {
            'symmetry': m1_data['symmetry'],
            'reference': m1_data['reference'],
            'ndoccpi': m1_data['ndoccpi'] + m2_data['ndoccpi'],
            'nsoccpi': m1_data['nsoccpi'] + m2_data['nsoccpi'],
            'nalphapi': m1_data['nalphapi'] + m2_data['nalphapi'],
            'nbetapi': m1_data['nbetapi'] + m2_data['nbetapi'],
            'BasisSet': m1_data['BasisSet'],
            'BasisSet PUREAM': m1_data['BasisSet PUREAM'],
        }

        data.update(d_Ca_occ.np_write(prefix='Ca_occ'))
        data.update(d_Cb_occ.np_write(prefix='Cb_occ'))
        m1_C_fn = "%s.%s.npz" % (core.get_writer_file_prefix(this_molecule.name()), constants.PSIF_SCF_MOS)
        np.savez(m1_C_fn, **data)

        core.set_local_option('SCF', 'GUESS', 'READ')

    def _init_df(self, calc):
        if self.no_reuse:
            return

        if calc.B == 'd':
            candidates = [calcid('m1', 'd', calc.Z), calcid('m2', 'd', calc.Z), calcid('d', 'd', calc.Z)]
            for c in filter(lambda c: c in self.wfn_cache, candidates):
                oldns = self.fmt_ns(c)
                newns = self.fmt_ns(calc)
                core.IO.change_file_namespace(constants.PSIF_DFSCF_BJ, oldns, newns)
                core.set_local_option("SCF", "DF_INTS_IO", "LOAD")
            else:
                core.set_local_option("SCF", "DF_INTS_IO", "SAVE")
        else:
            core.set_local_option("SCF", "DF_INTS_IO", "NONE")


    def compute(self, mol_name='m1', basis_center='m', basis_quality='low', mp2=False, mp2_dm=False):
        optstash = p4util.optproc.OptionsState(
            ['BASIS'],
            ['DF_BASIS_SCF'],
            ['DF_BASIS_MP2'],            
            ['SCF', 'GUESS'],
            ['SCF', 'DF_INTS_IO'],
            ['SCF_TYPE']
        )
        calc = calcid(mol_name, basis_center, basis_quality)
        self._init_ns(calc)
        self._init_df(calc)

        molecule = self.molecule(calc)
        basis = self.basis_sets[basis_quality]

        core.set_global_option('SCF_TYPE', 'DF')
        core.set_global_option('MP2_TYPE', 'DF')        
        core.set_global_option('BASIS', basis)
        core.set_global_option('DF_BASIS_SCF', basis + '-jkfit')
        core.set_global_option('DF_BASIS_MP2', basis + '-jkfit')        

        wfn = scf_helper('scf', molecule=molecule)
        if mp2 and not mp2_dm:
            wfn = run_dfmp2('df-mp2', molecule=molecule, ref_wfn=wfn)
        if mp2 and mp2_dm:
            optstash2 = p4util.optproc.OptionsState(
                ['ONEPDM'],
            )
            core.set_global_option("ONEPDM", True)
            wfn = run_dfmp2_gradient('df-mp2', molecule=molecule, ref_wfn=wfn)
            optstash2.restore()


        self.wfn_cache[calc] = wfn

        optstash.restore()
        return wfn



def run_many(name, molecule, **kwargs):
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


    c = WavefunctionCache(molecule, low='cc-pvdz', high='cc-pvtz')
    m1mhigh = c.compute('m1', 'm', 'high', mp2=True, mp2_dm=True) 
    m2mhigh = c.compute('m2', 'm', 'high', mp2=True, mp2_dm=True) 
    # m1dlow = c.compute('m1', 'd', 'low',  mp2=True) 
    # m2dlow = c.compute('m2', 'd', 'low',  mp2=True) 
    # m1dhigh = c.compute('m1', 'd', 'high', mp2=True)
    # m2dhigh = c.compute('m2', 'd', 'high', mp2=True)
    # ddlow = c.compute('d', 'd',  'low',  mp2=True)  
    ddhigh = c.compute('d', 'd',  'high', mp2=True) 

    def format_intene():
        def interaction(field, basis):
            d = c.wfn_cache[calcid('d', 'd', basis)]
            m1 = c.wfn_cache[calcid('m1', 'd', basis)]
            m2 = c.wfn_cache[calcid('m2', 'd', basis)]
            return d.get_variable(field) - (m1.get_variable(field) + m2.get_variable(field))

        low = os.path.basename(ddlow.basisset().name()).replace('.gbs', '')
        high = os.path.basename(ddhigh.basisset().name()).replace('.gbs', '')

        return {
            ('DF-MP2/%s CP Interaction Energy' % high):               interaction('MP2 TOTAL ENERGY', basis='high'),
            ('DF-HF/%s CP Interaction Energy' % high):                interaction('SCF TOTAL ENERGY', basis='high'),
            ('DF-MP2/%s CP Same-Spin Interaction Energy' % high):     interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='high'),
            ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % high): interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='high'),

            ('DF-MP2/%s CP Interaction Energy' % low):                interaction('MP2 TOTAL ENERGY', basis='low'),
            ('DF-HF/%s CP Interaction Energy' % low):                 interaction('SCF TOTAL ENERGY', basis='low'),
            ('DF-MP2/%s CP Same-Spin Interaction Energy' % low):      interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='low'),
            ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % low):  interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='low'),
        }

    def format_espx():
        decomp = EnergyDecomposition(monomer1=m1mhigh.molecule(), monomer2=m1mhigh.molecule(), dimer=ddhigh.molecule())
        with timer('Electrostatics_Overlap_Initialization'):
            esovlpfunc = decomp.esovlp()
        with timer('Electrostatics_Overal_Evaluation'):
            eshf, ovlhf = esovlpfunc(m1mhigh.reference_wavefunction(), m2mhigh.reference_wavefunction())
            esmp, ovlmp = esovlpfunc(m1mhigh, m2mhigh)

        # release some memory
        del esovlpfunc

        with timer('Heitler_London_Initialization'):
            hlfunc = decomp.hl()
        with timer('Heitler_London_Evaluation'):
            hl = hlfunc(m1mhigh, m1mhigh)

        basis = os.path.basename(m1mhigh.basisset().name()).replace('.gbs', '')
        return {
            'DF-HF/{} Electrostatic Interaction Energy'.format(basis): eshf,
            'DF-HF/{} Density Matrix Overlap'.format(basis): ovlhf,
            'DF-MP2/{} Electrostatic Interaction Energy'.format(basis): esmp,
            'DF-MP2{} Density Matrix Overlap'.format(basis): ovlmp,
            '{} Heitler-London Energy'.format(basis): hl
        }

    print(format_espx())
    # print(format_intene())


###############################################################################################

def run_nnmp2(name, molecule, **kwargs):
    return(run_many(name, molecule))


    optstash = p4util.optproc.OptionsState(
        ['SCF_TYPE'],
        ['MP2_TYPE'],
    )
    core.set_global_option('SCF_TYPE', 'DF')
    core.set_global_option('MP2_TYPE', 'DF')


    LOWBASIS = 'cc-pvdz'
    HIGHBASIS = 'cc-pvtz'

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

    espx_fields, m1hf_high, m2hf_high = run_espx(name, molecule=molecule, basis=HIGHBASIS)
    intene_fields, dimerhf_high = run_intene(name, molecule=molecule, lowbasis=LOWBASIS, highbasis=HIGHBASIS)
    sapt_fields = run_sapt(name, dimer_wfn=dimerhf_high, monomerA_wfn=m1hf_high, monomerB_wfn=m2hf_high, basis=HIGHBASIS)


    outlines = [
        '',
        '-' * 77,
        '=' * 24 + ' DESRES ENERGY DECOMPOSITION ' + '=' * 24,
        '-' * 77,
    ]

    for k, v in itertools.chain(
            sorted(espx_fields.iteritems()),
            sorted(intene_fields.iteritems()),
            sorted(sapt_fields.iteritems())):
        outlines.append('{:<52s} {:24.16f}'.format(k + ':', v))

    outlines.extend(['-' * 77, ''])
    core.print_out('\n'.join(outlines))
    core.tstop()

    optstash.restore()


def run_intene(name, molecule, lowbasis='aug-cc-pvtz', highbasis='aug-cc-pvqz'):
    monomer1, monomer2 = dimerize(molecule, basis='dimer')
    with timer('m1/dcbs'):
        m1low, m1high = run_dual_basis_mp2(name, molecule=monomer1, lowbasis=lowbasis, highbasis=highbasis)
    with timer('m2/dcbs'):
        m2low, m2high = run_dual_basis_mp2(name, molecule=monomer2, lowbasis=lowbasis, highbasis=highbasis)
    with timer('d/dcbs'):
        dlow, dhigh = run_dual_basis_mp2(name, molecule, lowbasis=lowbasis, highbasis=highbasis)

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
        ('DF-MP2/%s CP Interaction Energy' % highbasis):               difference('MP2 TOTAL ENERGY', basis='high'),
        ('DF-HF/%s CP Interaction Energy' % highbasis):                difference('SCF TOTAL ENERGY', basis='high'),
        ('DF-MP2/%s CP Same-Spin Interaction Energy' % highbasis):     difference('MP2 SAME-SPIN CORRELATION ENERGY', basis='high'),
        ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % highbasis): difference('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='high'),

        ('DF-MP2/%s CP Interaction Energy' % lowbasis):                difference('MP2 TOTAL ENERGY', basis='low'),
        ('DF-HF/%s CP Interaction Energy' % lowbasis):                 difference('SCF TOTAL ENERGY', basis='low'),
        ('DF-MP2/%s CP Same-Spin Interaction Energy' % lowbasis):      difference('MP2 SAME-SPIN CORRELATION ENERGY', basis='low'),
        ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % lowbasis):  difference('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='low'),
    }
    return value, dhigh.reference_wavefunction()


def run_espx(name, molecule, basis='aug-cc-pvqz'):
    optstash = p4util.optproc.OptionsState(
        ['BASIS'],
        ['DF_BASIS_MP2'],
        ['DF_BASIS_SCF'],
    )
    core.set_global_option("BASIS", basis)
    core.set_global_option("DF_BASIS_MP2", basis + '-jkfit')
    core.set_global_option("DF_BASIS_SCF", basis + '-jkfit')

    monomer1, monomer2 = dimerize(molecule, basis='monomer')

    with timer('Monomer_A'):
        m1hf, m1mp2 = run_mp2_density_matrix(basis=basis, molecule=monomer1)
    with timer('Monomer_B'):
        m2hf, m2mp2 = run_mp2_density_matrix(basis=basis, molecule=monomer2)

    with timer('Decomposition_Setup'):
        decomp = EnergyDecomposition(monomer1, monomer2, molecule)

    with timer('Electrostatics_Overlap_Initialization'):
        esovlpfunc = decomp.esovlp()
    with timer('Electrostatics_Overal_Evaluation'):
        eshf, ovlhf = esovlpfunc(m1hf, m2hf)
        esmp, ovlmp = esovlpfunc(m1mp2, m2mp2)

        # release some memory
        del esovlpfunc

    with timer('Heitler_London_Initialization'):
        hlfunc = decomp.hl()
    with timer('Heitler_London_Evaluation'):
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
    return values, m1hf, m2hf


def run_sapt(name, dimer_wfn, monomerA_wfn, monomerB_wfn, basis):
    optstash = p4util.optproc.OptionsState(
        ['SAPT', 'SAPT_LEVEL'],
        ['SAPT', 'E_CONVERGENCE'],
        ['SAPT', 'D_CONVERGENCE'],
        ['DF_BASIS_SAPT'],
        ['DF_BASIS_ELST'],
        ['BASIS'],
    )

    core.set_local_option('SCF', 'SCF_TYPE', 'DF')
    core.set_local_option('SAPT', 'SAPT_LEVEL', 'SAPT0')
    core.set_local_option('SAPT', 'E_CONVERGENCE', 10e-10)
    core.set_local_option('SAPT', 'D_CONVERGENCE', 10e-10)
    core.set_global_option('BASIS', basis)
    core.set_global_option('DF_BASIS_SAPT', basis + '-jkfit')

    core.IO.set_default_namespace('dimer')

    aux_basis = core.BasisSet.build(dimer_wfn.molecule(), "DF_BASIS_SAPT",
                                    core.get_global_option("DF_BASIS_SAPT"),
                                    "RIFIT", core.get_global_option("BASIS"))
    dimer_wfn.set_basisset("DF_BASIS_SAPT", aux_basis)
    dimer_wfn.set_basisset("DF_BASIS_ELST", aux_basis)
    e_sapt = core.sapt(dimer_wfn, monomerA_wfn, monomerB_wfn)

    optstash.restore()
    return {k: core.get_variable(k) for k in ('SAPT ELST10,R ENERGY', 'SAPT EXCH10 ENERGY',
            'SAPT EXCH10(S^2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY',
            'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY', 'SAPT SAME-SPIN EXCH-DISP20 ENERGY',
            'SAPT SAME-SPIN EXCH-DISP20 ENERGY', 'SAPT HF TOTAL ENERGY',
        )}




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
            Ca.np[:nbf1, :mol1_wfn.nalpha()] = mol1_wfn.Ca_subset('SO', 'OCC')
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


def run_mp2_density_matrix(molecule, basis, ref_wfn=None):
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

    optstash.restore()
    return ref_wfn, wfn_dfmp2



def run_dual_basis_mp2(name, molecule, lowbasis='aug-cc-pvtz', highbasis='aug-cc-pvqz'):
    optstash = p4util.optproc.OptionsState(
        ['BASIS'],
        ['SCF', 'GUESS'],
        ['DF_BASIS_SCF'],
        ['DF_BASIS_MP2'],
    )

    core.print_out('\n')
    p4util.banner(molecule.name())
    core.print_out('\n')

    guesspace = molecule.name() + '.low'
    core.IO.set_default_namespace(guesspace)

    # Run MP2 in the low basis set
    core.set_global_option('BASIS', lowbasis)
    core.set_global_option('DF_BASIS_SCF', lowbasis + '-jkfit')
    core.set_global_option('DF_BASIS_MP2', lowbasis + '-jkfit')
    low_wfn = run_dfmp2(name, molecule=molecule)

    if lowbasis == highbasis:
        return low_wfn, low_wfn

    # Move files to proper namespace
    namespace = molecule.name() + '.high'
    core.IO.change_file_namespace(180, guesspace, namespace)
    core.IO.set_default_namespace(namespace)

    # Run MP2 in the high basis set, using the low basis
    # result to acceperate convergence
    core.set_global_option('BASIS', highbasis)
    core.set_global_option('DF_BASIS_SCF', highbasis + '-jkfit')
    core.set_global_option('DF_BASIS_MP2', highbasis + '-jkfit')
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
psi4.driver.procedures['energy']['many'] = run_many
