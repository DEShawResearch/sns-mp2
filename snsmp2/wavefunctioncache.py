#                       SNS-MP2 LICENSE AGREEMENT
#
# Copyright 2017, D. E. Shaw Research. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#     this list of conditions, and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions, and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
# Neither the name of D. E. Shaw Research nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import shutil
import itertools
import numpy as np
import time
from collections import namedtuple
from psi4 import core
from psi4.driver.procrouting.proc import run_dfmp2, run_dfmp2_gradient
from psi4.driver.procrouting.proc import scf_helper
from psi4.driver.molutil import constants
from psi4 import extras
import psi4.driver.p4util as p4util
from .optstash import psiopts
from .frozencore import nfrozen_core


# For the WavefunctionCache, a calculation is identified by this three-tuple.
# V : The potential. This refers to the non-ghosted atoms. It is either
#     'm1', 'm2', or 'd'.
# B : The basis set location. This refers to the set of non-ghosted and ghosted
#     atoms. It is either 'm1', 'm2', or 'd'
# V : The basis set quality (i.e. zeta level). It's either 'low' or 'high'.
calcid = namedtuple('calcid', ('V', 'B', 'Z'))


class WavefunctionCache(object):
    """The purpose of this WavefunctionCache class is to manage and accelerate
    SCF calculations for a two-fragment system in which we will perform calculations
    on the dimer and the two monomers in both the dimer-centered basis set and the
    two monomer-centered basis sets. This cache also tries to help manage perfoming
    all of the calculations mentioned above in each of two different basis set zeta
    levels.

    These calculations can be accelerated a bit by using a couple tricks:

        - if we've already done a certain calculation in a small basis set, and now
          need to do the same calculation in a larger basis set, we can "upcast". 
        - if we've done two monomer calculations in the dimer basis set, we can
          "stack" the two wavefunctions to form a guess for the dimer calculation
          in the dimer basis set.
        - if we've done a monomer calculation in a monomer basis set and now
          want to do it in the dimer basis set, we can seed it from the converged
          monomer calculation.

    """
    def __init__(self, dimer, no_reuse=False, low='aug-cc-pvtz', high='aug-cc-pvtz'):
        self._original_path = os.path.abspath(os.curdir)
        self._d = dimer
        self._m1d, self._m2d = dimerize(dimer, basis='dimer')
        self._m1m, self._m2m = dimerize(dimer, basis='monomer')
        self.no_reuse = no_reuse
        self.wfn_cache = {}
        self.basis_sets = {
            'low': low,
            'high': high,
        }
        core.IOManager.shared_object().set_specific_retention(constants.PSIF_DFSCF_BJ, True)
        os.chdir(core.IOManager.shared_object().get_default_path())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):

        for calc in (calcid(*x) for x in itertools.product(('m1', 'm2', 'd'), ('m', 'd'), ('low', 'high'))):
            core.IO.set_default_namespace(self.fmt_ns(calc))
            core.IOManager.shared_object().set_specific_retention(constants.PSIF_DFSCF_BJ, False)
        try:
            extras.clean_numpy_files()
        except OSError:
            pass
        extras.numpy_files = []
        core.clean()
        os.chdir(self._original_path)

    def molecule(self, calc):
        # type: (calcid,) -> Molecule
        # Get the molecule object for the given calc.
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
        # type: (calcid,) -> str
        # Each calculation needs a namespace, which Psi4 core uses in the naming of the
        # temporary files it creates.
        return '%s-%s-%s' % (calc.V, calc.B, calc.Z)

    def _init_ns(self, calc):
        # type: (calcid,)
        # Move into new namespace, and create the relevant wavefunction guesses in
        # the new namespace by copying files over from old namespaces and modifying
        # them according to our heuristics (upcasting, stacking, adding ghosts)
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
        # type: (calcid, calcid)
        # Initialize a new namespace for `calc` with an opcast from `oldcalc`.
        assert oldcalc.V == calc.V and oldcalc.B == calc.B
        core.set_local_option('SCF', 'GUESS', 'READ')
        new_data = self._basis_projection(oldcalc, calc)
        newfn = self._fmt_mo_fn(calc)
        np.savez(newfn, **new_data)

        extras.register_numpy_file(newfn)
 
    def _basis_projection(self, oldcalc, newcalc):
        # There's a bug in Psi4 upcasting between custom basis sets
        # https://github.com/psi4/psi4/issues/719, so we do it ourselves.
        start_time = time.time()
        assert (oldcalc.B, oldcalc.Z) != (newcalc.B, newcalc.Z)

        read_filename = self._fmt_mo_fn(oldcalc)
        data = np.load(read_filename)
        Ca_occ = core.Matrix.np_read(data, "Ca_occ")
        Cb_occ = core.Matrix.np_read(data, "Cb_occ")
        puream = int(data["BasisSet PUREAM"])

        old_molecule = self.molecule(oldcalc)
        with psiopts('BASIS %s' % self.basis_sets[oldcalc.Z]):
            old_basis = core.BasisSet.build(old_molecule, "ORBITAL", self.basis_sets[oldcalc.Z], puream=puream)
            if isinstance(old_basis, tuple) and len(old_basis) == 2:
                # newer versions of psi return a second ECP basis
                old_basis = old_basis[0]

        new_molecule = self.molecule(newcalc)
        with psiopts('BASIS %s' % self.basis_sets[newcalc.Z]):
            new_basis = core.BasisSet.build(new_molecule, 'ORBITAL', self.basis_sets[newcalc.Z], puream=puream)

            if isinstance(new_basis, tuple) and len(new_basis) == 2:
                # newer versions of psi return a second ECP basis
                base_wfn = core.Wavefunction(new_molecule, *new_basis)
                new_basis = new_basis[0]
            else:
                base_wfn = core.Wavefunction(new_molecule, new_basis)

        nalphapi = core.Dimension.from_list(data["nalphapi"])
        nbetapi = core.Dimension.from_list(data["nbetapi"])
        pCa = base_wfn.basis_projection(Ca_occ, nalphapi, old_basis, new_basis)
        pCb = base_wfn.basis_projection(Cb_occ, nbetapi, old_basis, new_basis)

        new_data = {}
        new_data.update(pCa.np_write(None, prefix="Ca_occ"))
        new_data.update(pCb.np_write(None, prefix="Cb_occ"))
        new_data["reference"] = core.get_option('SCF', 'REFERENCE')
        new_data["symmetry"] = new_molecule.schoenflies_symbol()
        new_data["BasisSet"] = new_basis.name()
        new_data["BasisSet PUREAM"] = puream

        core.print_out('\n Computing basis set projection from {calc1} to {calc2} (elapsed={time:.2f})\n'.format(
            calc1=self._display_name(oldcalc).lower(),
            calc2=self._display_name(newcalc).lower(),
            time=time.time()-start_time,
        ))

        # Workaround for https://github.com/psi4/psi4/pull/750
        for key, value in new_data.items():
            if isinstance(value, np.ndarray) and value.flags['OWNDATA'] == False:
                new_data[key] = np.copy(value)

        return new_data

    def _fmt_mo_fn(self, calc):
        # type: (calcid,) -> str
        # Path to the molecular orbital file for a calc.
        return "%s.%s.npz" % (core.get_writer_file_prefix(self.fmt_ns(calc)), constants.PSIF_SCF_MOS)

    def _init_addghost_C(self, oldcalc, calc):
        # print('Adding ghost %s->%s' % (oldcalc, calc))

        old_filename = self._fmt_mo_fn(oldcalc)
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

        write_filename = core.get_writer_file_prefix(self.fmt_ns(calc)) + ".180.npz"
        np.savez(write_filename, **data_dict)
        extras.register_numpy_file(write_filename)
        core.set_local_option('SCF', 'GUESS', 'READ')

    def _init_stack_C(self, calc, oldcalc_m1, oldcalc_m2):
        assert oldcalc_m1.V == 'm1'
        assert oldcalc_m2.V == 'm2'
        # print('Stacking monomer wfns', calc, oldcalc_m1, oldcalc_m2)

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
            core.set_global_option("DF_INTS_IO", "SAVE")
            for c in filter(lambda c: c in self.wfn_cache, candidates):
                oldns = self.fmt_ns(c)
                newns = self.fmt_ns(calc)
                core.IO.change_file_namespace(constants.PSIF_DFSCF_BJ, oldns, newns)
                core.set_global_option("DF_INTS_IO", "LOAD")
        else:
            core.set_global_option("DF_INTS_IO", "NONE")

    def _banner(self, calc, mp2=False, mp2_dm=False):
        # type: (calcid, Optional[bool], Optional[bool])
        core.print_out('\n')
        p4util.banner(' Scheduling {fmt_calc} ({c}) '.format(
            fmt_calc=self._display_name(calc),
            c='MP2 density matrix' if mp2_dm else ('MP2' if mp2 else 'HF')))
        core.print_out('\n')

    def _display_name(self, calc):
        # type: (calcid) -> string
        mol_name, basis_center, basis_quality = calc
        mol_name = {'m1': 'Monomer A', 'm2': 'Monomer B', 'd': 'Dimer'}[mol_name]
        centered = {'m': 'monomer', 'd': 'dimer'}[basis_center]
        quality = self.basis_sets[basis_quality]

        return '{mol_name} in {quality} {centered}-centered basis'.format(
            mol_name=mol_name, quality=quality, centered=centered)

    def compute(self, mol_name='m1', basis_center='m', basis_quality='low', mp2=False, mp2_dm=False, save_jk=False):
        calc = calcid(mol_name, basis_center, basis_quality)
        molecule = self.molecule(calc)
        molecule.set_name(self.fmt_ns(calc))
        basis = self.basis_sets[basis_quality]

        self._banner(calc, mp2, mp2_dm)

        optstash = p4util.optproc.OptionsState(
            ['SCF', 'DF_INTS_IO'],
            ['DF_INTS_IO'],
            ['SCF', 'GUESS'])
        self._init_ns(calc)
        self._init_df(calc)

        with psiopts(
                'SCF_TYPE DF',
                'MP2_TYPE DF',
                'BASIS %s' % basis,
                'DF_BASIS_SCF %s-JKFIT' % basis,
                'DF_BASIS_MP2 %s-RI' % basis,
                'SCF SAVE_JK %s' % save_jk,
                'ONEPDM TRUE',
                'NUM_FROZEN_DOCC %d' % nfrozen_core(molecule),
                ):

            wfn = scf_helper('scf', molecule=molecule)
            assert nfrozen_core(molecule) == wfn.nfrzc()

            if mp2 and not mp2_dm:
                wfn = run_dfmp2('df-mp2', molecule=molecule, ref_wfn=wfn)
            if mp2 and mp2_dm:
                wfn = run_dfmp2_gradient('df-mp2', molecule=molecule, ref_wfn=wfn)
            if mp2_dm and not mp2:
                raise ValueError('These options dont make sense')

        self.wfn_cache[calc] = wfn
        optstash.restore()
        core.clean()
        return wfn


def dimerize(molecule, basis='monomer'):
    nfrag = molecule.nfragments()
    if nfrag != 2:
        raise ValueError('NN-MP2 requires active molecule to have 2 fragments, not %s.' % (nfrag))

    if basis == 'monomer':
        monomer1 = molecule.extract_subsets(1)
        monomer2 = molecule.extract_subsets(2)
    elif basis == 'dimer':
        monomer1 = molecule.extract_subsets(1, 2)
        monomer2 = molecule.extract_subsets(2, 1)

    return monomer1, monomer2

