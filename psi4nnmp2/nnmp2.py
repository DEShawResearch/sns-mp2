import itertools
import json
from psi4 import core
import psi4.driver.p4util as p4util
import time

from .eshlovlp import ESHLOVLPDecomposition
from .wavefunctioncache import WavefunctionCache
from .format_output import format_espx_dict, format_intene_dict, format_sapt0_dict
from .format_output import format_intene_human, format_espx_human
from .optstash import psiopts
from .resources import vminfo

# DEBUG = True



@psiopts('freeze_core desresval')
def run_nnmp2(name, molecule, do_sapt=True, do_espx=True, do_intene=True, **kwargs):
    """Run the NN-MP2 calculation

    """
    if len(kwargs) > 0:
        raise ValueError('Unrecognized options: %s' % str(kwargs))
    core.tstart()
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
    if do_sapt:
        m1mlow  = c.compute('m1', 'm', 'low', mp2=False)
        m2mlow  = c.compute('m2', 'm', 'low', mp2=False)
    if do_espx:
        m1mhigh = c.compute('m1', 'm', 'high', mp2=True, mp2_dm=True)
        m2mhigh = c.compute('m2', 'm', 'high', mp2=True, mp2_dm=True)
    if do_intene:
        m1dlow  = c.compute('m1', 'd', 'low',  mp2=True)
        m2dlow  = c.compute('m2', 'd', 'low',  mp2=True)
        m1dhigh = c.compute('m1', 'd', 'high', mp2=True)
        m2dhigh = c.compute('m2', 'd', 'high', mp2=True)
    if do_sapt and (not do_intene):
        ddlow   = c.compute('d', 'd',  'low',  mp2=False)
    if do_intene:
        ddlow = c.compute('d', 'd',  'low',  mp2=True)
        ddhigh  = c.compute('d', 'd',  'high', mp2=True)

    @psiopts(
        'SCF_TYPE DF',
        # If we previously did the intene, we can reuse
        # the DF_INTS
        'DF_INTS_IO %s' % ('LOAD' if do_intene else 'NONE'),
        'BASIS %s' % HIGH,
    )
    def run_espx():
        core.tstart()
        p4util.banner('ESPX')

        if do_intene:
            dimer_basis = ddhigh.basisset()
        else:
            dimer_basis = core.BasisSet.build(molecule)

        aux_basis = core.BasisSet.build(molecule, "DF_BASIS_SCF",
                                        HIGH+'-jkfit', "JKFIT", HIGH)

        decomp = ESHLOVLPDecomposition(
            m1_basis=m1mhigh.basisset(),
            m2_basis=m2mhigh.basisset(),
            dimer_basis=dimer_basis,
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
        }, dimer_basis

    ###################################################################

    @psiopts(
        'SAPT SAPT_LEVEL SAPT0',
        'SAPT E_CONVERGENCE 10e-10',
        'SAPT D_CONVERGENCE 10e-10',
        'SAPT NAT_ORBS_T2 TRUE',
        'SCF_TYPE DF',
        'BASIS %s' % LOW)
    def run_sapt():
        if ddlow.name() == 'SCF':
            dimer_wfn = ddlow
        else:
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

    ###################################################################


    # Run the three previously defined functions
    json_data = []
    human_data = []

    if do_espx:
        espx_data, dimer_high_basis = run_espx()
        json_data.extend(format_espx_dict(molecule, dimer_high_basis, espx_data))
        human_data.extend(sorted(format_espx_human(HIGH, espx_data).iteritems()))
    if do_sapt:
        sapt_data = run_sapt()
        json_data.append(format_sapt0_dict(ddlow, sapt_data))
        human_data.extend(sorted(sapt_data.iteritems()))
    if do_intene:
        json_data.append(format_intene_dict(m1dlow, m2dlow, ddlow))
        json_data.append(format_intene_dict(m1dhigh, m2dhigh, ddhigh))
        human_data.extend(sorted(format_intene_human(c).iteritems()))

    outlines = [
        '',
        '-' * 80,
        '=' * 22 + ' DESRES ENERGY DECOMPOSITION (JSON) ' + '=' * 22,
        '-' * 80,
    ]
    outlines.append(json.dumps(json_data))
    outlines.extend(['-' * 80, '', ''])
    core.print_out('\n'.join(outlines))


    outlines = [
        '',
        '-' * 86,
        '=' * 28 + '  DESRES ENERGY DECOMPOSITION ' + '=' * 28,
        '-' * 28 + '            (a.u)             ' + '-' * 28,
        '-' * 86,
    ]
    for k, v in human_data:
        outlines.append('{:<60s} {:24.16f}'.format(k + ':', v))
    outlines.extend(['-' * 86, ''])

    core.print_out('\n'.join(outlines))
    core.tstop()

    for k, v in human_data:
        core.set_variable(k, v)


    core.print_out('''
  ---------------------
  ==> NN-MP2 Memory <==
  ---------------------
  VmHWM:   %.2f MB

''' % vminfo()['VmHWM'])
