import itertools
from collections import namedtuple
import numpy as np
from psi4 import core
import psi4.driver.p4util as p4util

from .eshlovlp import ESHLOVLPDecomposition
from .wavefunctioncache import WavefunctionCache, calcid


def run_nnmp2(name, molecule, **kwargs):
    # Force to c1
    molecule = molecule.clone()
    molecule.reset_point_group('c1')
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    nfrag = molecule.nfragments()
    if nfrag != 2:
        raise ValidationError('NN-MP2 requires active molecule to have 2 fragments, not %s.' % (nfrag))

    LOW = 'aug-cc-pvtz'
    HIGH = 'aug-cc-pvqz'

    c = WavefunctionCache(molecule, low=LOW, high=HIGH)
    m1mlow  = c.compute('m1', 'm', 'low', mp2=False)
    m2mlow  = c.compute('m2', 'm', 'low', mp2=False)
    m1mhigh = c.compute('m1', 'm', 'high', mp2=True, mp2_dm=True) 
    m2mhigh = c.compute('m2', 'm', 'high', mp2=True, mp2_dm=True) 
    m1dlow  = c.compute('m1', 'd', 'low',  mp2=True)
    m2dlow  = c.compute('m2', 'd', 'low',  mp2=True) 
    m1dhigh = c.compute('m1', 'd', 'high', mp2=True)
    m2dhigh = c.compute('m2', 'd', 'high', mp2=True)
    ddlow   = c.compute('d', 'd',  'low',  mp2=True)  
    ddhigh  = c.compute('d', 'd',  'high', mp2=True) 

    ###################################################################

    def format_intene():
        def interaction(field, basis):
            d = c.wfn_cache[calcid('d', 'd', basis)]
            m1 = c.wfn_cache[calcid('m1', 'd', basis)]
            m2 = c.wfn_cache[calcid('m2', 'd', basis)]
            return d.get_variable(field) - (m1.get_variable(field) + m2.get_variable(field))

        return {
            ('DF-MP2/%s CP Interaction Energy' % HIGH):
                interaction('MP2 TOTAL ENERGY', basis='high'),
            ('DF-HF/%s CP Interaction Energy' % HIGH):
                interaction('SCF TOTAL ENERGY', basis='high'),
            ('DF-MP2/%s CP Same-Spin Interaction Energy' % HIGH):
                interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='high'),
            ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % HIGH):
                interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='high'),

            ('DF-MP2/%s CP Interaction Energy' % LOW):
                interaction('MP2 TOTAL ENERGY', basis='low'),
            ('DF-HF/%s CP Interaction Energy' % LOW):
                interaction('SCF TOTAL ENERGY', basis='low'),
            ('DF-MP2/%s CP Same-Spin Interaction Energy' % LOW):
                interaction('MP2 SAME-SPIN CORRELATION ENERGY', basis='low'),
            ('DF-MP2/%s CP Opposite-Spin Interaction Energy' % LOW):
                interaction('MP2 OPPOSITE-SPIN CORRELATION ENERGY', basis='low'),
        }

    ###################################################################

    def format_espx():
        optstash = p4util.optproc.OptionsState(
            ['DF_INTS_IO'],
            ['SCF_TYPE'],
        )
        core.set_global_option('SCF_TYPE', 'DF')
        core.set_global_option('DF_INTS_IO', 'LOAD')
        aux_basis = core.BasisSet.build(molecule, "DF_BASIS_MP2",
                                        HIGH+'-jkfit', "RIFIT", HIGH)

        decomp = ESHLOVLPDecomposition(
            m1_basis=m1mhigh.basisset(),
            m2_basis=m2mhigh.basisset(),
            dimer_basis=ddhigh.basisset(),
            dimer_aux_basis=aux_basis)

        esovlpfunc = decomp.esovlp()
        eshf, ovlhf = esovlpfunc(m1mhigh.reference_wavefunction(), m2mhigh.reference_wavefunction())
        esmp, ovlmp = esovlpfunc(m1mhigh, m2mhigh)

        # release some memory
        del esovlpfunc

        hlfunc = decomp.hl()
        hl = hlfunc(m1mhigh.reference_wavefunction(), m2mhigh.reference_wavefunction())

        optstash.restore()
        return {
            'DF-HF/{} Electrostatic Interaction Energy'.format(HIGH): eshf,
            'DF-HF/{} Density Matrix Overlap'.format(HIGH): ovlhf,
            'DF-MP2/{} Electrostatic Interaction Energy'.format(HIGH): esmp,
            'DF-MP2/{} Density Matrix Overlap'.format(HIGH): ovlmp,
            'DF-HF/{} Heitler-London Energy'.format(HIGH): hl
        }

    ###################################################################

    def format_sapt():
        optstash = p4util.optproc.OptionsState(
            ['SAPT', 'SAPT_LEVEL'],
            ['SAPT', 'E_CONVERGENCE'],
            ['SAPT', 'D_CONVERGENCE'],
            ['DF_BASIS_SAPT'],
            ['DF_BASIS_ELST'],
            ['SAPT', 'NAT_ORBS_T2'],
            ['BASIS'],
        )

        core.set_local_option('SCF', 'SCF_TYPE', 'DF')
        core.set_local_option('SAPT', 'SAPT_LEVEL', 'SAPT0')
        core.set_local_option('SAPT', 'E_CONVERGENCE', 10e-10)
        core.set_local_option('SAPT', 'D_CONVERGENCE', 10e-10)
        core.set_local_option('SAPT', 'NAT_ORBS_T2', True)
        core.set_global_option('BASIS', LOW)
        core.set_global_option('DF_BASIS_SAPT', LOW + '-jkfit')

        dimer_wfn = ddlow.reference_wavefunction()
        aux_basis = core.BasisSet.build(dimer_wfn.molecule(), "DF_BASIS_SAPT",
                                        core.get_global_option("DF_BASIS_SAPT"),
                                        "RIFIT", core.get_global_option("BASIS"))
        dimer_wfn.set_basisset("DF_BASIS_SAPT", aux_basis)
        dimer_wfn.set_basisset("DF_BASIS_ELST", aux_basis)
        e_sapt = core.sapt(dimer_wfn, m1mlow, m2mlow)

        optstash.restore()
        return {k: core.get_variable(k) for k in ('SAPT ELST10,R ENERGY', 'SAPT EXCH10 ENERGY',
                'SAPT EXCH10(S^2) ENERGY', 'SAPT IND20,R ENERGY', 'SAPT EXCH-IND20,R ENERGY',
                'SAPT EXCH-DISP20 ENERGY', 'SAPT DISP20 ENERGY', 'SAPT SAME-SPIN EXCH-DISP20 ENERGY',
                'SAPT SAME-SPIN EXCH-DISP20 ENERGY', 'SAPT HF TOTAL ENERGY',
        )}

    ###################################################################

    core.tstart()

    # Run the three previously defined functions
    espx_fields = format_espx()
    intene_fields = format_intene()
    sapt_fields = format_sapt()

    # Format output
    outlines = [
        '',
        '-' * 77,
        '=' * 24 + ' DESRES ENERGY DECOMPOSITION ' + '=' * 24,
        '-' * 77,
    ]
    for k, v in itertools.chain(
            sorted(espx_fields.iteritems()),
            sorted(intene_fields.iteritems()),
            sorted(sapt_fields.iteritems())):
        outlines.append('{:<55s} {:24.16f}'.format(k + ':', v))

    outlines.extend(['-' * 77, ''])
    core.print_out('\n'.join(outlines))

    core.tstop()

