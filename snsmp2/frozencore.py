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


CORE_POLICY_VALENCE = {'Ni': 6, 'Sb': 4, 'Ru': 0, 'Na': 1, 'Nb': 0, 'Mg': 1, 'Li': 0, 'Y': 0, 'Pd': 0, 'Ti': 5, 'Te': 4, 'Rh': 0, 'Tc': 0, 'Be': 0, 'Xe': 4, 'Si': 5, 'As': 4, 'Fe': 6, 'Br': 4, 'Mo': 0, 'He': 0, 'C': 1, 'B': 1, 'F': 1, 'I': 4, 'H': 0, 'K': 5, 'Mn': 5, 'O': 1, 'Ne': 1, 'Q': 0, 'P': 5, 'S': 5, 'Kr': 4, 'V': 5, 'Sc': 5, 'X': 0, 'N': 1, 'Se': 4, 'Zn': 1, 'Co': 6, 'Ag': 1, 'Cl': 5, 'Ca': 5, 'Al': 2, 'Cd': 1, 'Ge': 4, 'Ar': 5, 'Zr': 0, 'Ga': 4, 'In': 1, 'Cr': 5, 'Cu': 1, 'Sn': 4}



def nfrozen_core(molecule):
    # Freeze according to orbital energy based "valence" policy

    # Unfortunately, the Psi4 molecule class does _not_ make it easy
    # to extract information about ghosts. But ghosts have Z() == 0,
    # so we can get the symbols of the non-ghost atoms this way.
    real_atoms = [molecule.symbol(i).title() for i in range(molecule.natom()) if molecule.Z(i) > 0]
    return sum(CORE_POLICY_VALENCE[e] for e in real_atoms)
