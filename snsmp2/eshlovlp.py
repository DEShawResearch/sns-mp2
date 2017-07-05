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



from psi4 import core
import numpy as np


class ESHLOVLPDecomposition(object):
    def __init__(self, m1_basis, m2_basis, dimer_basis, dimer_aux_basis):
        self.dimer_basis = dimer_basis
        self.dimer_aux_basis = dimer_aux_basis
        self.dimer_V, self.dimer_T, self.dimer_S = self._initialize_mints(dimer_basis)
        self.monomer1_V = self._initialize_mints(m1_basis, v_only=True)
        self.monomer2_V = self._initialize_mints(m2_basis, v_only=True)

        self.nuclear_interaction_energy = (
            dimer_basis.molecule().nuclear_repulsion_energy() -
            m1_basis.molecule().nuclear_repulsion_energy() -
            m2_basis.molecule().nuclear_repulsion_energy())

    def _initialize_jk(self, basis, aux_basis, do_J=True, do_K=True):
        jk = core.JK.build(basis, aux_basis)
        jk.set_memory(int(float(core.get_global_option("SCF_MEM_SAFETY_FACTOR")) * core.get_memory() / 8))
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
        C.np[:mol1_wfn.nso(), :mol1_wfn.nalpha()] = m1_OCC.np[:, :]
        C.np[-mol2_wfn.nso():, -mol2_wfn.nalpha():] = m2_OCC.np[:, :]

        C = orthogonalize(C, self.p.dimer_S)

        # At this point, it should be the case that
        # C.T * S * C == I
        np.testing.assert_array_almost_equal(
            core.Matrix.triplet(C, self.p.dimer_S, C, True, False, False),
            np.eye(nocc))

        self.jk.C_clear()
        self.jk.C_left_add(C)
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

        energy = FH.vector_dot(D) + self.p.dimer_basis.molecule().nuclear_repulsion_energy()
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

        U, S, VT = np.linalg.svd(mol1_wfn.Da())
        C_left = np.dot(U, np.diag(np.sqrt(S)))
        C_right = np.dot(VT.T, np.diag(np.sqrt(S)))

        C_left_ = core.Matrix(nbf, max(nbf1, nbf2))
        C_right_ = core.Matrix(nbf, max(nbf1, nbf2))

        C_left_.np[:nbf1, :nbf1] = C_left[:, :]
        C_right_.np[:nbf1, :nbf1] = C_right[:, :]

        self.jk.C_clear()
        self.jk.C_left_add(C_left_)
        self.jk.C_right_add(C_right_)
        self.jk.compute()

        J = self.jk.J()[0]
        D1 = self.jk.D()[0]
        assert np.max(np.abs(D1.np[:nbf1, :nbf1] - mol1_wfn.Da().np)) < 1e-10

        J_1to2 = J.np[nbf1:, nbf1:]
        elel_1to2 = 2 * np.sum(J_1to2 * mol2_wfn.Da())
        nuel_1to2 = 2 * (self.p.dimer_V.vector_dot(D1) - self.p.monomer1_V.vector_dot(mol1_wfn.Da()))
        ovlp1 = core.Matrix.doublet(self.p.dimer_S, D1, False, False)

        #######################################################################

        U, S, VT = np.linalg.svd(mol2_wfn.Da())
        C_left = np.dot(U, np.diag(np.sqrt(S)))
        C_right = np.dot(VT.T, np.diag(np.sqrt(S)))

        C_left_ = core.Matrix(nbf, max(nbf1, nbf2))
        C_right_ = core.Matrix(nbf, max(nbf1, nbf2))


        C_left_.np[-nbf2:, -nbf2:] = C_left[:, :]
        C_right_.np[-nbf2:, -nbf2:] = C_right[:, :]

        self.jk.C_clear()
        self.jk.C_left_add(C_left_)
        self.jk.C_right_add(C_right_)
        self.jk.compute()

        J = self.jk.J()[0]
        D2 = self.jk.D()[0]
        assert np.max(np.abs(D2.np[nbf1:, nbf1:] - mol2_wfn.Da().np)) < 1e-10

        J_2to1 = J.np[:nbf1, :nbf1]
        elel_2to1 = 2 * np.sum(J_2to1 * mol1_wfn.Da())
        nuel_2to1 = 2 * (self.p.dimer_V.vector_dot(D2) - self.p.monomer2_V.vector_dot(mol2_wfn.Da()))

        ovlp2 = core.Matrix.doublet(self.p.dimer_S, D2, False, False)

        overlap = 4 * np.sum(ovlp1.np * ovlp2.np.T)
        #assert abs(elel_1to2 - elel_2to1) < 1e-10

        electrostatic = self.p.nuclear_interaction_energy + nuel_1to2 + nuel_2to1 + elel_1to2 + elel_2to1
        return electrostatic, overlap




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
