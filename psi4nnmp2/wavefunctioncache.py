import os
import shutil
import numpy as np
from collections import namedtuple
from psi4 import core
from psi4.driver.procrouting.proc import run_dfmp2, run_dfmp2_gradient
from psi4.driver.procrouting.proc import scf_helper
from psi4.driver.molutil import constants
from psi4 import extras
import psi4.driver.p4util as p4util


calcid = namedtuple('calcid', ('V', 'B', 'Z'))


class WavefunctionCache(object):
    def __init__(self, dimer, no_reuse=False, low='aug-cc-pvtz', high='aug-cc-pvtz'):
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

        # print('nocast')

    def _init_upcast_C(self, oldcalc, calc):
        # print('Upcasting %s->%s' % (oldcalc, calc))
        assert oldcalc.V == calc.V and oldcalc.B == calc.B
        core.set_local_option('SCF', 'GUESS', 'READ')
        oldfn = self._fmt_mo_fn(oldcalc)
        newfn = self._fmt_mo_fn(calc)
        shutil.copy(oldfn, newfn)
        extras.register_numpy_file(newfn)

    def _fmt_mo_fn(self, calc):
        return "%s.%s.npz" % (core.get_writer_file_prefix(self.fmt_ns(calc)), constants.PSIF_SCF_MOS)

    def _init_addghost_C(self, oldcalc, calc):
        # print('Adding ghost %s->%s' % (oldcalc, calc))
        this_molecule = self.molecule(calc)
        old_molecule = self.molecule(oldcalc)

        old_filename = self._fmt_mo_fn(oldcalc)
        data = np.load(old_filename)
        Ca_occ = core.Matrix.np_read(data, "Ca_occ")
        Cb_occ = core.Matrix.np_read(data, "Cb_occ")
        #oldwfn = self.wfn_cache[oldcalc]
        #Ca_occ = oldwfn.Ca_subset('SO', 'OCC')
        #Cb_occ = oldwfn.Cb_subset('SO', 'OCC')

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

        write_filename = core.get_writer_file_prefix(self.fmt_ns(calc)) + ".180.npz"
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

        m1_C_fn = self._fmt_mo_fn(oldcalc_m1)
        m2_C_fn = self._fmt_mo_fn(oldcalc_m2)
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
        m1_C_fn = self._fmt_mo_fn(calc)
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


    def compute(self, mol_name='m1', basis_center='m', basis_quality='low', mp2=False, mp2_dm=False, save_jk=False):
        optstash = p4util.optproc.OptionsState(
            ['BASIS'],
            ['DF_BASIS_SCF'],
            ['DF_BASIS_MP2'],            
            ['SCF', 'GUESS'],
            ['SCF', 'DF_INTS_IO'],
            ['SCF_TYPE'],
            ['SCF', 'SAVE_JK']
        )
        calc = calcid(mol_name, basis_center, basis_quality)
        molecule = self.molecule(calc)
        molecule.set_name(self.fmt_ns(calc))
        basis = self.basis_sets[basis_quality]

        self._init_ns(calc)
        self._init_df(calc)

        core.set_global_option('SCF_TYPE', 'DF')
        core.set_global_option('MP2_TYPE', 'DF')        
        core.set_global_option('BASIS', basis)
        core.set_global_option('DF_BASIS_SCF', basis + '-jkfit')
        core.set_global_option('DF_BASIS_MP2', basis + '-jkfit')        
        core.set_local_option('SCF', 'SAVE_JK', save_jk)

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


def dimerize(molecule, basis='monomer'):
    if basis == 'monomer':
        monomer1 = molecule.extract_subsets(1)
        monomer2 = molecule.extract_subsets(2)
    elif basis == 'dimer':
        monomer1 = molecule.extract_subsets(1, 2)
        monomer2 = molecule.extract_subsets(2, 1)

    return monomer1, monomer2

