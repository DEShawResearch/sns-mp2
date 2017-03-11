import itertools
from collections import namedtuple
import numpy as np
import json
from psi4 import core
import psi4.driver.p4util as p4util

from .eshlovlp import ESHLOVLPDecomposition
from .wavefunctioncache import WavefunctionCache, calcid
from .format_output import format_espx_dict, format_intene_dict, format_sapt0_dict
from .format_output import format_intene_human, format_espx_human


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

    def run_espx():
        core.tstart()
        p4util.banner(' ESPX')
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

        del esovlpfunc

        hlfunc = decomp.hl()
        hl = hlfunc(m1mhigh.reference_wavefunction(), m2mhigh.reference_wavefunction())

        core.tstop()
        optstash.restore()
        return {
            'eshf': eshf,
            'ovlhf': ovlhf,
            'esmp': esmp,
            'ovlmp': ovlmp,
            'hl': hl
        }

    ###################################################################

    def run_sapt():
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
                'SAPT SAME-SPIN DISP20 ENERGY', 'SAPT HF TOTAL ENERGY',
        )}


    # Run the three previously defined functions
    espx_data = run_espx()
    sapt_data = run_sapt()

    for_json = [
        format_intene_dict(m1dlow, m2dlow, ddlow),
        format_intene_dict(m1dhigh, m2dhigh, ddhigh),
        format_sapt0_dict(ddlow, sapt_data),
    ]
    for_json.extend(format_espx_dict(ddhigh, espx_data))


    outlines = [
        '',
        '-' * 80,
        '=' * 22 + ' DESRES ENERGY DECOMPOSITION (JSON) ' + '=' * 22,
        '-' * 80,
    ]
    outlines.append(json.dumps(for_json))
    outlines.extend(['-' * 80, '', ''])
    core.print_out('\n'.join(outlines))


    outlines = [
        '',
        '-' * 80,
        '=' * 26 + ' DESRES ENERGY DECOMPOSITION ' + '=' * 25,
        '-' * 26 + '         (Hartrees)          ' + '-' * 25,
        '-' * 80,
    ]
    for k, v in itertools.chain(
            sorted(format_espx_human(HIGH, espx_data).iteritems()),
            sorted(format_intene_human(c).iteritems()),
            sorted(sapt_data.iteritems())):
        outlines.append('{:<55s} {:24.16f}'.format(k + ':', v))
    outlines.extend(['-' * 80, ''])

    core.print_out('\n'.join(outlines))
    core.tstop()

    output_dict = dict()
    output_dict.update(format_espx_human(HIGH, espx_data))
    output_dict.update(format_intene_human(c).iteritems())
    output_dict.update(sapt_data)
    for k, v in output_dict.items():
        core.set_variable(k, v)