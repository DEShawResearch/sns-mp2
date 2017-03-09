import os
import itertools
import numpy as np
import psi4
from psi4 import core
import psi4.driver.p4util as p4util
from psi4.driver.procrouting.proc import run_dfmp2, run_dfmp2_gradient
from psi4.driver.procrouting.proc import scf_helper
from psi4.driver.molutil import constants
from psi4 import extras
from decomp2 import EnergyDecomposition
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

    def _init_upcast_C(self, oldcalc, calc):
        # print('Upcasting', oldcalc, calc)
        assert oldcalc.V == calc.V and oldcalc.B == calc.B
        core.set_local_option('SCF', 'GUESS', 'READ')
        core.IO.change_file_namespace(constants.PSIF_SCF_MOS, self.fmt_ns(oldcalc), self.fmt_ns(calc))

    def _init_addghost_C(self, oldcalc, calc):
        # print('Adding ghost', oldcalc, calc)
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
        # print('Stacking monomer wfns', calc, oldcalc_m1, oldcalc_m2)
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



def run_nnmp2(name, molecule, **kwargs):
    # Force to c1
    molecule = molecule.clone()
    molecule.reset_point_group('c1')
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    nfrag = molecule.nfragments()
    if nfrag != 2:
        raise ValidationError('NN-MP2 requires active molecule to have 2 fragments, not %s.' % (nfrag))

    LOW = 'cc-pvdz'
    HIGH = 'cc-pvdz'

    # Run the 8 HF and MP2 calculations we need
    c = WavefunctionCache(molecule, low=LOW, high=HIGH)
    m1mhigh = c.compute('m1', 'm', 'high', mp2=True, mp2_dm=True) 
    m2mhigh = c.compute('m2', 'm', 'high', mp2=True, mp2_dm=True) 
    m1dlow  = c.compute('m1', 'd', 'low',  mp2=True) 
    m2dlow  = c.compute('m2', 'd', 'low',  mp2=True) 
    m1dhigh = c.compute('m1', 'd', 'high', mp2=True)
    m2dhigh = c.compute('m2', 'd', 'high', mp2=True)
    ddlow   = c.compute('d', 'd',  'low',  mp2=True)  
    ddhigh  = c.compute('d', 'd',  'high', mp2=True) 

    ###################################################################

    def format_intene():
        def interaction(field, basis):
            d = c.wfn_cache[calcid('d', 'd', basis)]
            m1 = c.wfn_cache[calcid('m1', 'd', basis)]
            m2 = c.wfn_cache[calcid('m2', 'd', basis)]
            return d.get_variable(field) - (m1.get_variable(field) + m2.get_variable(field))

        return {
            ('DF-MP2/%s CP Interaction Energy' % HIGH):               interaction('MP2 TOTAL ENERGY', basis='high'),
            ('DF-HF/%s CP Interaction Energy' % HIGH):                interaction('SCF TOTAL ENERGY', basis='high'),
            ('DF-MP2/%s CP Same-Spin Interaction Energy' % HIGH):     interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='high'),
            ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % HIGH): interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='high'),

            ('DF-MP2/%s CP Interaction Energy' % LOW):                interaction('MP2 TOTAL ENERGY', basis='low'),
            ('DF-HF/%s CP Interaction Energy' % LOW):                 interaction('SCF TOTAL ENERGY', basis='low'),
            ('DF-MP2/%s CP Same-Spin Interaction Energy' % LOW):      interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='low'),
            ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % LOW):  interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='low'),
        }

    ###################################################################

    def format_espx():
        aux_basis = core.BasisSet.build(ddhigh.molecule(), "DF_BASIS_MP2",
                                        HIGH+'-jkfit', "RIFIT", HIGH)
        optstash = p4util.optproc.OptionsState(
            ['DF_INTS_IO'],
            ['SCF_TYPE']
        )
        core.set_global_option('SCF_TYPE', 'DF')
        core.set_global_option('DF_INTS_IO', 'LOAD')

        decomp = EnergyDecomposition(
            m1_basis=m1mhigh.basisset(),
            m2_basis=m2mhigh.basisset(),
            dimer_basis=ddhigh.basisset(),
            dimer_aux_basis=aux_basis)

        esovlpfunc = decomp.esovlp()
        eshf, ovlhf = esovlpfunc(m1mhigh.reference_wavefunction(), m2mhigh.reference_wavefunction())
        esmp, ovlmp = esovlpfunc(m1mhigh, m2mhigh)

        # release some memory
        del esovlpfunc

        hlfunc = decomp.hl()
        hl = hlfunc(m1mhigh, m1mhigh)

        optstash.restore()
        return {
            'DF-HF/{} Electrostatic Interaction Energy'.format(HIGH): eshf,
            'DF-HF/{} Density Matrix Overlap'.format(HIGH): ovlhf,
            'DF-MP2/{} Electrostatic Interaction Energy'.format(HIGH): esmp,
            'DF-MP2{} Density Matrix Overlap'.format(HIGH): ovlmp,
            '{} Heitler-London Energy'.format(HIGH): hl
        }

    ###################################################################

    def format_sapt():
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
        core.set_global_option('BASIS', HIGH)
        core.set_global_option('DF_BASIS_SAPT', HIGH + '-jkfit')

        dimer_wfn = ddhigh.reference_wavefunction()
        aux_basis = core.BasisSet.build(dimer_wfn.molecule(), "DF_BASIS_SAPT",
                                        core.get_global_option("DF_BASIS_SAPT"),
                                        "RIFIT", core.get_global_option("BASIS"))
        dimer_wfn.set_basisset("DF_BASIS_SAPT", aux_basis)
        dimer_wfn.set_basisset("DF_BASIS_ELST", aux_basis)
        e_sapt = core.sapt(dimer_wfn, m1mhigh.reference_wavefunction(), m2mhigh.reference_wavefunction())

        optstash.restore()
        return {k: core.get_variable(k) for k in ('SAPT ELST10,R ENERGY', 'SAPT EXCH10 ENERGY',
                'SAPT EXCH10(S^2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY',
                'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY', 'SAPT SAME-SPIN EXCH-DISP20 ENERGY',
                'SAPT SAME-SPIN EXCH-DISP20 ENERGY', 'SAPT HF TOTAL ENERGY',
        )}

    ###################################################################

    core.tstart()

    # Run the three previously defined functions
    espx_fields = format_espx()
    intene_fields = format_intene()
    sapt_fields = format_sapt()

    # Format output
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


psi4.driver.procedures['energy']['nnmp2'] = run_nnmp2
