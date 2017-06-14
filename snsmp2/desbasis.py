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
