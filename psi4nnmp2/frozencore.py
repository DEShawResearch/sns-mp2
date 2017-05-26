CORE_POLICY_VALENCE = {'Ni': 6, 'Sb': 4, 'Ru': 0, 'Na': 1, 'Nb': 0, 'Mg': 1, 'Li': 0, 'Y': 0, 'Pd': 0, 'Ti': 5, 'Te': 4, 'Rh': 0, 'Tc': 0, 'Be': 0, 'Xe': 4, 'Si': 5, 'As': 4, 'Fe': 6, 'Br': 4, 'Mo': 0, 'He': 0, 'C': 1, 'B': 1, 'F': 1, 'I': 4, 'H': 0, 'K': 5, 'Mn': 5, 'O': 1, 'Ne': 1, 'Q': 0, 'P': 5, 'S': 5, 'Kr': 4, 'V': 5, 'Sc': 5, 'X': 0, 'N': 1, 'Se': 4, 'Zn': 1, 'Co': 6, 'Ag': 1, 'Cl': 5, 'Ca': 5, 'Al': 2, 'Cd': 1, 'Ge': 4, 'Ar': 5, 'Zr': 0, 'Ga': 4, 'In': 1, 'Cr': 5, 'Cu': 1, 'Sn': 4}



def nfrozen_core(molecule):
    # Freeze according to orbital energy based "valence" policy

    # Unfortunately, the Psi4 molecule class does _not_ make it easy
    # to extract information about ghosts. But ghosts have Z() == 0,
    # so we can get the symbols of the non-ghost atoms this way.
    real_atoms = [molecule.symbol(i).title() for i in range(molecule.natom()) if molecule.Z(i) > 0]
    return sum(CORE_POLICY_VALENCE[e] for e in real_atoms)
