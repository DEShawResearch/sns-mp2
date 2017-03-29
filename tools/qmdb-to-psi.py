'''
This script queries CCSD(T) data in QMDB under the {'raw/ccsd_nmer2', 'raw/ccsd_ligand_nmer_solv',
'raw/ccsd_nmer_solv2', 'raw/eccc_ligand'} project names, and generates Psi4 input files
using the psi4nnmp2 plugin to calculate raw MP2 input quantities for the neural network.
'''
from __future__ import print_function, division
import garden
garden.load('ffde-monolith/0.2.14-st010/lib-python')

import os
import csv
import sys
import time
import qmdb
import jinja2
import operator
import msys
import argparse
from pprint import pformat
import itertools
import pandas as pd
from collections import Counter


TEMPLATE = jinja2.Template('''generated = '{{now}}'
frame_metadata = {{metadata}}
refvals = {{refvals}}

molecule {
{% for frag in fragments -%}
{{frag['charge']}} {{frag['spin_multiplicity']}}
{% for i in frag['atoms'] -%}
{{elements[i]}} {{'{:14.10f}'.format(xyz[i,0])}} {{'{:14.10f}'.format(xyz[i,1])}} {{'{:14.10f}'.format(xyz[i,2])}}
{% endfor -%}
{% if not loop.last %}--
{% endif -%}
{% endfor %}
symmetry c1
no_com
no_reorient
}

set memory 7gb

import psi4nnmp2
energy('nnmp2', do_intene={{missing['intene']}}, do_espx={{missing['espx']}}, do_sapt={{missing['psi4_sapt0']}})
''')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('outdir', help='Directory in which to write output files')
    p.add_argument('--oldindex', type=pd.read_csv)
    args = p.parse_args()
    execute(args, p)


def execute(args, p):
    db = qmdb.QMDBPrototype(host='qmdb', version=time.strftime("%Y-%m-%d %H:%M"), port=6432)  # QMDB server in EN
    scans = db.get_scans(project_name__in=(
        'raw/ccsd_nmer2',
        'raw/ccsd_ligand_nmer_solv',
        'raw/ccsd_nmer_solv2',
        'raw/eccc_ligand',
    ), calculation_type='intene', theory='CCSD(T)', raw=True)


    frameiter = itertools.chain.from_iterable(
        (scan.get_frame(i) for i in range(len(scan)))
        for scan in scans)

    olddf = args.oldindex

    count = itertools.count()
    with open('index.csv', 'w') as indx:
        csvw =  csv.writer(indx, dialect='excel', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(['index', 'system_name', 'frame_id', 'scan_id', 'system_id', 'effsize', 'filename', 'do_espx', 'do_sapt', 'do_intene'])

        for frame in frameiter:
            frame_id = frame.frame_id
            refvals, work_to_do = generate_refvals(frame)
            work_did = {
                k.replace('do_', '').replace('sapt', 'psi4_sapt0'): v
                for k, v in olddf[olddf.frame_id == frame_id].squeeze()[['do_espx', 'do_intene', 'do_sapt']].to_dict().iteritems()}
            new_work_to_do = {
                k: (work_to_do[k] and (not work_did[k]))
            for k in ('espx', 'intene', 'psi4_sapt0')}

            work_to_do = new_work_to_do

            try:
                p4in, metadata = generate_psi4in(frame, refvals=refvals, work_to_do=work_to_do)
            except ElementToHeavyError:
                print('Too heavy! Skipping %s' % frame.scan.system.name)
                continue


            fn = os.path.join(args.outdir, metadata['system_name'], metadata['scan_name'], 
                '%s-%s-%s.in' % (metadata['system_name'], metadata['scan_name'], metadata['frame_index']))

            if (not work_to_do['espx']) and (not work_to_do['psi4_sapt0']) and (not work_to_do['intene']):
                print('No work for %s' % fn)
                continue

            i = next(count)
            writefile(fn, p4in)
            csvw.writerow([i, metadata['system_name'], metadata['frame_id'], metadata['scan_id'],
                           metadata['system_id'], metadata['effsize'], fn, work_to_do['espx'],
                           work_to_do['psi4_sapt0'], work_to_do['intene']])

            if i % 10 == 0:
                print('\r', i, end=' ')
                sys.stdout.flush()
                indx.flush()




def generate_refvals(frame):
    def mp2_interaction(scan):
        return scan.dimer_total_energy - (scan.monomer0_total_energy + scan.monomer1_total_energy)
    def hf_interaction(scan):
        return scan.dimer_reference_energy - (scan.monomer0_reference_energy + scan.monomer1_reference_energy)

    fields = {}
    have = {'intene': False, 'espx': False, 'espx': False, 'psi4_sapt0': False}

    for f in frame.get_same_geometry_frames():
        s = f.scan

        if s.calculation_type == None and s.basis_set == None:
            continue

        try:
            basis = {'desavqz-rev5': 'desavqz-psi-rev1', 
                     'desavtz-rev5': 'desavtz-psi-rev1',
                     'desavtz-psi-rev1': 'desavtz-psi-rev1',
                     None: None}[s.basis_set]
        except KeyError:
            continue

        # the six sets are:
        # mp2 tz intene, mp2 qz intene, hf qz espx, mp2 qz espx, sapt0

        if s.calculation_type == 'intene':
            fields['DF-MP2/{} CP Interaction Energy'.format(basis)] = (mp2_interaction(s)[f.frame_index], f.frame_id)
            fields['DF-HF/{} CP Interaction Energy'.format(basis)] = (hf_interaction(s)[f.frame_index], f.frame_id)
            have['mp2/intene/%s' % basis]  = True

        elif s.calculation_type == 'espx' and s.theory == 'HF':
            fields['DF-HF/{} Density Matrix Overlap'.format(basis)] = (s.OVL[f.frame_index], f.frame_id)
            fields['DF-HF/{} Electrostatic Interaction Energy'.format(basis)] = (s.ES[f.frame_index], f.frame_id)
            fields['DF-HF/{} Heitler-London Energy'.format(basis)] = (s.HL[f.frame_index], f.frame_id)
            have['hf/espx/%s' % basis] = True

        elif s.calculation_type == 'espx' and s.theory == 'MP2':
            fields['DF-MP2/{} Density Matrix Overlap'.format(basis)] = (s.OVL[f.frame_index], f.frame_id)
            fields['DF-MP2/{} Electrostatic Interaction Energy'.format(basis)] = (s.ES[f.frame_index], f.frame_id)
            have['mp2/espx/%s' % basis] = True

        elif s.calculation_type == 'psi4_sapt0' and s.basis_set == 'desavtz-psi-rev1':
            fields['SAPT ELST10,R ENERGY'] = (getattr(s, 'Elst10,r')[f.frame_index], f.frame_id)
            fields['SAPT EXCH10 ENERGY'] = (getattr(s, 'Exch10')[f.frame_index], f.frame_id)
            fields['SAPT EXCH10(S^2) ENERGY'] = (getattr(s, 'Exch10(S^2)')[f.frame_index], f.frame_id)
            fields['SAPT IND20,R ENERGY'] = (getattr(s, 'Ind20,r')[f.frame_index], f.frame_id)
            fields['SAPT EXCH-IND20,R ENERGY'] = (getattr(s, 'Exch-Ind20,r')[f.frame_index], f.frame_id)
            fields['SAPT EXCH-DISP20 ENERGY'] = (getattr(s, 'Exch-Disp20')[f.frame_index], f.frame_id)
            fields['SAPT DISP20 ENERGY'] = (getattr(s, 'Disp20')[f.frame_index], f.frame_id)
            fields['SAPT SAME-SPIN EXCH-DISP20 ENERGY'] = (getattr(s, 'Exch-Disp20 (SS)')[f.frame_index], f.frame_id)
            fields['SAPT SAME-SPIN DISP20 ENERGY'] = (getattr(s, 'Disp20 (SS)')[f.frame_index], f.frame_id)
            have['psi4_sapt0'] = True
        elif s.calculation_type == 'espx' and s.theory == 'CCSD(T)':
            continue
        else:
            print(s.calculation_type, s.theory, s.basis_set)

    missing = {
        'psi4_sapt0': not have['psi4_sapt0'],
        'espx': not (have.get('hf/espx/desavqz-psi-rev1', False) and have.get('mp2/espx/desavqz-psi-rev1', False)),
        'intene': not (have.get('mp2/intene/desavqz-psi-rev1', False) and have.get('hf/intene/desavqz-psi-rev1', False) and
                       have.get('mp2/intene/desavtz-psi-rev1', False) and have.get('hf/intene/desavtz-psi-rev1', False))
    }
    return fields, missing


def generate_psi4in(frame, refvals=None, work_to_do=None):
    element_types = frame.scan.metadata['element_types']
    if contains_above_nickel(element_types):
        raise ElementToHeavyError()

    if refvals is None or work_to_do is None:
        refvals, work_to_do = generate_refvals(frame)

    metadata = dict(
            project_name=frame.scan.project_name,
            scan_id=frame.scan.id,
            scan_name=frame.scan.name,
            system_id=frame.scan.system_id,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            effsize=effsize(element_types),
    )
    rendered_p4in = TEMPLATE.render(
        now=time.strftime("%Y-%m-%d %H:%M"),
        fragments=frame.scan.metadata['fragments'],
        elements=frame.scan.metadata['element_types'],
        xyz=frame.scan.xyz[frame.frame_index],
        metadata=pformat(metadata, width=10),
        refvals=pformat({k: {'energy': v[0], 'frame_id': v[1]} for k, v in refvals.items()}),
        missing=work_to_do,
    )
    metadata['system_name'] = frame.scan.system.name

    return rendered_p4in, metadata


def effsize(element_types):
    size = 0
    for s in (msys.ElementForAbbreviation(str(t)) for t in element_types):
        if s <= 2:
            size += 1
        elif s <= 10:
            size += 2
        elif s <= 18:
            size += 3
        elif s <= 36:
            size += 4
        elif s <= 54:
            size += 5
        else:
            raise ValueError()
    return size


def contains_above_nickel(element_types):
    '''Psi4 doesn't support ECPs, which we want for
    elements Cu and above. So we're going to skip.
    '''
    for s in (msys.ElementForAbbreviation(str(t)) for t in element_types):
        if s > 28:
            return True
    return False


def makedir(fn):
    try:
        os.makedirs(os.path.dirname(fn))
    except OSError:
        pass


def writefile(fn, contents):
    makedir(fn)
    with open(fn, 'wb') as f:
        f.write(contents)
        if not contents.endswith('\n'):
            f.write('\n')


class ElementToHeavyError(Exception):
    pass


if __name__ == '__main__':
    main()