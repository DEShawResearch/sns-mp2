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
import numpy as np
from psi4 import core
from psi4.driver.molutil import constants
from .wavefunctioncache import dimerize, calcid


def format_intene_human(wfncache):
    def interaction(field, basis):
        d = wfncache.wfn_cache[calcid('d', 'd', basis)]
        m1 = wfncache.wfn_cache[calcid('m1', 'd', basis)]
        m2 = wfncache.wfn_cache[calcid('m2', 'd', basis)]
        return d.get_variable(field) - (m1.get_variable(field) + m2.get_variable(field))
    return {
        ('DF-MP2/%s CP Interaction Energy' % wfncache.basis_sets['high']):
            interaction('MP2 TOTAL ENERGY', basis='high'),
        ('DF-HF/%s CP Interaction Energy' % wfncache.basis_sets['high']):
            interaction('SCF TOTAL ENERGY', basis='high'),
        ('DF-MP2/%s CP Same-Spin Interaction Energy' % wfncache.basis_sets['high']):
            interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='high'),
        ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % wfncache.basis_sets['high']):
            interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='high'),

        ('DF-MP2/%s CP Interaction Energy' % wfncache.basis_sets['low']):
            interaction('MP2 TOTAL ENERGY', basis='low'),
        ('DF-HF/%s CP Interaction Energy' % wfncache.basis_sets['low']):
            interaction('SCF TOTAL ENERGY', basis='low'),
        ('DF-MP2/%s CP Same-Spin Interaction Energy' % wfncache.basis_sets['low']):
            interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='low'),
        ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % wfncache.basis_sets['low']):
            interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='low'),
    }


def format_espx_human(basis, espx_data):
    return {
        'DF-HF/{} Electrostatic Interaction Energy'.format(basis): espx_data['eshf'],
        'DF-HF/{} Density Matrix Overlap'.format(basis): espx_data['ovlhf'],
        'DF-MP2/{} Electrostatic Interaction Energy'.format(basis): espx_data['esmp'],
        'DF-MP2/{} Density Matrix Overlap'.format(basis): espx_data['ovlmp'],
        'DF-HF/{} Heitler-London Energy'.format(basis): espx_data['hl'],
    }


# def format_sapt0_dict(dwfn, sapt_data):

#     dHF2 = sapt_data['SAPT HF TOTAL ENERGY'] - (
#         sapt_data['SAPT ELST10,R ENERGY'] + 
#         sapt_data['SAPT EXCH10 ENERGY'] + 
#         sapt_data['SAPT IND20,R ENERGY'] + 
#         sapt_data['SAPT EXCH-IND20,R ENERGY'])
#     tot_ind = (sapt_data['SAPT IND20,R ENERGY'] + 
#                dHF2 +
#                sapt_data['SAPT EXCH-IND20,R ENERGY'])

#     input = _format_input_block_dict(dwfn)
#     input['method'] = 'SAPT0'

#     return {
#         'calculation_type': 'sapt0',
#         'error': False,
#         'filename': os.path.abspath(core.outfile_name()),
#         'input': input,
#         'output': {
#             'Disp20': sapt_data['SAPT DISP20 ENERGY'],
#             'Disp20 (OS)': sapt_data['SAPT DISP20 ENERGY'] - sapt_data['SAPT SAME-SPIN DISP20 ENERGY'],
#             'Disp20 (SS)': sapt_data['SAPT SAME-SPIN DISP20 ENERGY'],
#             'Exch-Disp20': sapt_data['SAPT EXCH-DISP20 ENERGY'],
#             'Exch-Disp20 (SS)': sapt_data['SAPT SAME-SPIN EXCH-DISP20 ENERGY'],
#             'Exch-Disp20 (OS)': sapt_data['SAPT EXCH-DISP20 ENERGY'] - sapt_data['SAPT SAME-SPIN EXCH-DISP20 ENERGY'],            
#             'Dispersion': sapt_data['SAPT DISP20 ENERGY'] + sapt_data['SAPT EXCH-DISP20 ENERGY'],
#             'Electrostatics': sapt_data['SAPT ELST10,R ENERGY'],
#             'Elst10,r': sapt_data['SAPT ELST10,R ENERGY'],
#             'Exch10': sapt_data['SAPT EXCH10 ENERGY'],
#             'Exch10(S^2)': sapt_data['SAPT EXCH10(S^2) ENERGY'],
#             'Exchange': sapt_data['SAPT EXCH10 ENERGY'],
#             'Ind20,r': sapt_data['SAPT IND20,R ENERGY'],
#             'Exch-Ind20,r': sapt_data['SAPT EXCH-IND20,R ENERGY'],
#             'delta HF,r (2)': dHF2,
#             'Induction': tot_ind,
#         }
#     }


def format_espx_dict(molecule, basis, espx_data):

    class DummyHFWfn(object):
        @staticmethod
        def name():
            return 'SCF'

        @staticmethod
        def basisset():
            return basis

        @staticmethod
        def molecule():
            return molecule

    class DummyMP2Wfn(DummyHFWfn):
        @staticmethod
        def name():
            return 'DF-MP2'

    return [
        {
            'calculation_type': 'espx',
            'error': False,
            'filename': os.path.abspath(core.outfile_name()),
            'input': _format_input_block_dict(DummyMP2Wfn()),
            'output': {
                'ES': espx_data['esmp'],
                'OVL': espx_data['ovlmp'],
            }
        },
        {
            'calculation_type': 'espx',
            'error': False,
            'filename': os.path.abspath(core.outfile_name()),
            'input': _format_input_block_dict(DummyHFWfn),
            'output': {
                'ES': espx_data['eshf'],
                'OVL': espx_data['ovlhf'],
                'HL': espx_data['hl'],
            }
        },
    ]


def format_intene_dict(m1wfn, m2wfn, dwfn):
    return {
        'calculation_type': 'intene',
        'error': False,
        'filename': os.path.abspath(core.outfile_name()),
        'input': _format_input_block_dict(dwfn),
        'output': {
            'dimer': {
                'reference_energy': dwfn.get_variable('SCF TOTAL ENERGY'),
                'correlation_energy': dwfn.get_variable('MP2 CORRELATION ENERGY'),
                'singlet_pair_energy': dwfn.get_variable('MP2 OPPOSITE-SPIN CORRELATION ENERGY'),
                'triplet_pair_energy': dwfn.get_variable('MP2 SAME-SPIN CORRELATION ENERGY'),
                'total_energy': dwfn.get_variable('MP2 TOTAL ENERGY'),
            },
            'monomers': [
                {
                    'reference_energy': m1wfn.get_variable('SCF TOTAL ENERGY'),
                    'correlation_energy': m1wfn.get_variable('MP2 CORRELATION ENERGY'),
                    'singlet_pair_energy': m1wfn.get_variable('MP2 OPPOSITE-SPIN CORRELATION ENERGY'),
                    'triplet_pair_energy': m1wfn.get_variable('MP2 SAME-SPIN CORRELATION ENERGY'),
                    'total_energy': m1wfn.get_variable('MP2 TOTAL ENERGY'),
                },
                {
                    'reference_energy': m2wfn.get_variable('SCF TOTAL ENERGY'),
                    'correlation_energy': m2wfn.get_variable('MP2 CORRELATION ENERGY'),
                    'singlet_pair_energy': m2wfn.get_variable('MP2 OPPOSITE-SPIN CORRELATION ENERGY'),
                    'triplet_pair_energy': m2wfn.get_variable('MP2 SAME-SPIN CORRELATION ENERGY'),
                    'total_energy': m2wfn.get_variable('MP2 TOTAL ENERGY'),
                },
            ]
        }
    }



def _format_input_block_dict(dimer_wfn=None):
    molecule = dimer_wfn.molecule()
    s1, s2 = dimerize(molecule, basis='monomer')

    return {
        'element_types': [molecule.label(i) for i in range(molecule.natom())],
        'charge': molecule.molecular_charge(),
        'spin_multiplicity': molecule.multiplicity(),
        'basis_set': os.path.splitext(os.path.basename(dimer_wfn.basisset().name()))[0],
        'method': {'DF-MP2': 'df-mp2', 'SCF': 'hf'}[dimer_wfn.name()],
        'title': '',
        'xyz': (np.asarray(molecule.geometry()) * constants.bohr2angstroms).tolist(),
        'fragments': [
            {'atoms': list(range(s1.natom())),
             'charge': s1.molecular_charge(),
             'spin_multiplicity': s1.multiplicity()
            },
            {'atoms': list(range(s1.natom(), molecule.natom())),
             'charge': s2.molecular_charge(),
             'spin_multiplicity': s2.multiplicity()
            }
        ],

    }
