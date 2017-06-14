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
