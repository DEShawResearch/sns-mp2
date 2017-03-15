import itertools
import json
from psi4 import core
import psi4.driver.p4util as p4util

from .eshlovlp import ESHLOVLPDecomposition
from .wavefunctioncache import WavefunctionCache
from .format_output import format_espx_dict, format_intene_dict, format_sapt0_dict
from .format_output import format_intene_human, format_espx_human
from .optstash import psiopts

# DEBUG = True


@psiopts('freeze_core desresval')
def run_nnmp2(name, molecule, **kwargs):
    # Force to c1
    molecule = molecule.clone()
    molecule.reset_point_group('c1')
    molecule.fix_orientation(True)
    molecule.fix_com(True)
    molecule.update_geometry()

    nfrag = molecule.nfragments()
    if nfrag != 2:
        raise ValueError('NN-MP2 requires active molecule to have 2 fragments, not %s.' % (nfrag))

    LOW = 'desavtz-psi-rev1'
    HIGH = 'desavqz-psi-rev1'
    #if DEBUG:
    #    LOW = 'desvdz-psi-rev1'
    #    HIGH = 'desvtz-psi-rev1'

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


    @psiopts(
        'SCF_TYPE DF',
        'DF_INTS_IO LOAD'
    )
    def run_espx():
        core.tstart()
        p4util.banner('ESPX')
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
        return {
            'eshf': eshf,
            'ovlhf': ovlhf,
            'esmp': esmp,
            'ovlmp': ovlmp,
            'hl': hl
        }

    ###################################################################

    @psiopts(
        'SAPT SAPT_LEVEL SAPT0',
        'SAPT E_CONVERGENCE 10e-10',
        'SAPT D_CONVERGENCE 10e-10',
        'SAPT NAT_ORBS_T2 TRUE',
        'SCF_TYPE DF',
        'DF_BASIS_SAPT %s-jkfit' % LOW,
        'BASIS %s' % LOW)
    def run_sapt():
        dimer_wfn = ddlow.reference_wavefunction()
        aux_basis = core.BasisSet.build(dimer_wfn.molecule(), "DF_BASIS_SAPT",
                                        core.get_global_option("DF_BASIS_SAPT"),
                                        "RIFIT", core.get_global_option("BASIS"))
        dimer_wfn.set_basisset("DF_BASIS_SAPT", aux_basis)
        dimer_wfn.set_basisset("DF_BASIS_ELST", aux_basis)
        core.sapt(dimer_wfn, m1mlow, m2mlow)
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
        '-' * 82,
        '=' * 27 + ' DESRES ENERGY DECOMPOSITION ' + '=' * 27,
        '-' * 27 + '           (a.u)             ' + '-' * 27,
        '-' * 82,
    ]
    for k, v in itertools.chain(
            sorted(format_espx_human(HIGH, espx_data).iteritems()),
            sorted(format_intene_human(c).iteritems()),
            sorted(sapt_data.iteritems())):
        outlines.append('{:<57s} {:24.16f}'.format(k + ':', v))
    outlines.extend(['-' * 82, ''])

    core.print_out('\n'.join(outlines))
    core.tstop()

    output_dict = dict()
    output_dict.update(format_espx_human(HIGH, espx_data))
    output_dict.update(format_intene_human(c).iteritems())
    output_dict.update(sapt_data)
    for k, v in output_dict.items():
        core.set_variable(k, v)