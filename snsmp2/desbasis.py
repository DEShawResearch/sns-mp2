import json
import os.path

from psi4.driver import qcdb
from psi4 import core
from psi4.driver.qcdb import periodictable


def inject_desres_basis():
    with open(os.path.join(os.path.dirname(__file__), 'desbasis.json')) as f:
        basis_json = json.load(f)

    for name, spans in basis_json.items():
        anon = _build_basis_funtion(name, spans)
        anon.__name__ = 'basisspec_psi4_yo__%s' % str(name).replace('-', '')
        qcdb.libmintsbasisset.basishorde[name.upper()] = anon


def _build_basis_funtion(name, spans):
    def anon(mol, role):
        for span, target in spans.items():
            start, end = span.split('-')
            for z in range(int(start), int(end)+1):
                symbol = periodictable.z2el[z]
                mol.set_basis_by_symbol(symbol, str(target), role=role)
    return anon
