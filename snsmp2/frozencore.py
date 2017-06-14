# Copyright 2017  D. E. Shaw Research, LLC
# 
# All rights reserved.
# 
# D. E. Shaw Research, LLC ("DESRES") hereby grants you a limited,
# revocable license to use and/or modify the attached computer code (the
# "Code") for internal purposes only. Redistribution of any kind and in
# any form is strictly prohibited.
# 
# In consideration of the rights granted to you hereunder, you hereby
# agree to indemnify, defend and hold DESRES harmless from and against
# any and all damages sustained by DESRES, which damages arise out of or
# relate to your use of the Code, including without limitation any
# damages caused by any allegation or claim that the Code infringes the
# rights of any third party.
# 
# THE ATTACHED CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL DESRES BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS CODE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


CORE_POLICY_VALENCE = {'Ni': 6, 'Sb': 4, 'Ru': 0, 'Na': 1, 'Nb': 0, 'Mg': 1, 'Li': 0, 'Y': 0, 'Pd': 0, 'Ti': 5, 'Te': 4, 'Rh': 0, 'Tc': 0, 'Be': 0, 'Xe': 4, 'Si': 5, 'As': 4, 'Fe': 6, 'Br': 4, 'Mo': 0, 'He': 0, 'C': 1, 'B': 1, 'F': 1, 'I': 4, 'H': 0, 'K': 5, 'Mn': 5, 'O': 1, 'Ne': 1, 'Q': 0, 'P': 5, 'S': 5, 'Kr': 4, 'V': 5, 'Sc': 5, 'X': 0, 'N': 1, 'Se': 4, 'Zn': 1, 'Co': 6, 'Ag': 1, 'Cl': 5, 'Ca': 5, 'Al': 2, 'Cd': 1, 'Ge': 4, 'Ar': 5, 'Zr': 0, 'Ga': 4, 'In': 1, 'Cr': 5, 'Cu': 1, 'Sn': 4}



def nfrozen_core(molecule):
    # Freeze according to orbital energy based "valence" policy

    # Unfortunately, the Psi4 molecule class does _not_ make it easy
    # to extract information about ghosts. But ghosts have Z() == 0,
    # so we can get the symbols of the non-ghost atoms this way.
    real_atoms = [molecule.symbol(i).title() for i in range(molecule.natom()) if molecule.Z(i) > 0]
    return sum(CORE_POLICY_VALENCE[e] for e in real_atoms)
