from __future__ import print_function, division
import os
import sys
import time
import qmdb
import jinja2
import operator
import msys
import argparse
from pprint import pformat
import itertools
from collections import Counter


def main():
    db = qmdb.QMDBPrototype(host='qmdb', port=6432)  # QMDB server in EN
    scans = db.get_scans(project_name__in=(
        'raw/ccsd_nmer2',
        'raw/ccsd_ligand_nmer_solv',
        'raw/ccsd_nmer_solv2',
        'raw/eccc_ligand',
    ), calculation_type='intene', theory='CCSD(T)', raw=True)


    frameiter = itertools.chain.from_iterable(
        (scan.get_frame(i) for i in range(len(scan)))
        for scan in scans)

    import csv
    with open('index.csv', 'w') as indx:
        csvw =  csv.writer(indx, dialect='excel', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(['index', 'system_name', 'frame_id', 'scan_id', 'system_id', 'effsize', 'filename', 'do_espx', 'do_sapt', 'do_intene'])

        for i, frame in enumerate(frameiter):
            p4in, metadata, work_to_do = generate_psi4in(frame)

            fn = os.path.join(metadata['system_name'], metadata['scan_name'], 
                '%s-%s-%s.in' % (metadata['system_name'], metadata['scan_name'], metadata['frame_index']))

            try:
                os.makedirs(os.path.dirname(fn))
            except OSError:
                pass
            with open(fn, 'w') as f:
                f.write(p4in)

            csvw.writerow([i, metadata['system_name'], metadata['frame_id'], metadata['scan_id'],
                           metadata['system_id'], metadata['effsize'], fn, work_to_do['espx'],
                           work_to_do['psi4_sapt0'], work_to_do['intene']])
            indx.flush()

            print('\r', i, end=' ')
            sys.stdout.flush()


def generate_refvals(frame):
    def mp2_interaction(scan):
        return scan.dimer_total_energy - (scan.monomer0_total_energy + scan.monomer1_total_energy)
    def hf_interaction(scan):
        return scan.dimer_reference_energy - (scan.monomer0_reference_energy + scan.monomer1_reference_energy)
    def update_resources(f):
        r = f.get_metadata()['resources_used']
        resources_used['cpu_time'] += r.get('cpu_time', 0.0)
        resources_used['real_time'] += r.get('real_time', 0)
        resources_used['disk'] = max(resources_used['disk'], r.get('disk', 0))
        resources_used['memory'] = max(resources_used['memory'], r.get('memory', 0))

    resources_used = {
        'cpu_time': 0,
        'disk': 0,
        'memory': 0,
        'real_time': 0,
    }

    fields = {}
    have = {'intene': False, 'mp2espx': False, 'hfespx': False, 'psi4_sapt0': False}

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


        if s.calculation_type == 'intene':
            fields['DF-MP2/{} CP Interaction Energy'.format(basis)] = (mp2_interaction(s)[f.frame_index], f.frame_id)
            fields['DF-HF/{} CP Interaction Energy'.format(basis)] = (hf_interaction(s)[f.frame_index], f.frame_id)
            update_resources(f)
            have['intene'] = True

        elif s.calculation_type == 'espx' and s.theory == 'HF':
            fields['DF-HF/{} Density Matrix Overlap'.format(basis)] = (s.OVL[f.frame_index], f.frame_id)
            fields['DF-HF/{} Electrostatic Interaction Energy'.format(basis)] = (s.ES[f.frame_index], f.frame_id)
            fields['DF-HF/{} Heitler-London Energy'.format(basis)] = (s.HL[f.frame_index], f.frame_id)
            update_resources(f)
            have['hfespx'] = True

        elif s.calculation_type == 'espx' and s.theory == 'MP2':
            fields['DF-MP2/{} Density Matrix Overlap'.format(basis)] = (s.OVL[f.frame_index], f.frame_id)
            fields['DF-MP2/{} Electrostatic Interaction Energy'.format(basis)] = (s.ES[f.frame_index], f.frame_id)
            update_resources(f)
            have['mp2espx'] = True

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
            update_resources(f)
            have['psi4_sapt0'] = True
        else:
            print(s.calculation_type, s.theory, s.basis_set)

    have['espx'] = (have.pop('hfespx') and have.pop('mp2espx'))
    missing = {k: not have[k] for k in have.keys()}

    return fields, resources_used, missing


def generate_psi4in(frame):
    t = jinja2.Template('''generated = '{{now}}'
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

    refvals, resources, missing = generate_refvals(frame)
    metadata = dict(
            project_name=frame.scan.project_name,
            scan_id=frame.scan.id,
            scan_name=frame.scan.name,
            system_name=frame.scan.system.name,
            system_id=frame.scan.system_id,
            frame_id=frame.frame_id,
            frame_index=frame.frame_index,
            effsize=effsize(frame.scan.metadata['element_types']),
    )
    return t.render(
        now=time.strftime("%Y-%m-%d %H:%M"),
        fragments=frame.scan.metadata['fragments'],
        elements=frame.scan.metadata['element_types'],
        xyz=frame.scan.xyz[frame.frame_index],
        metadata=pformat(metadata, width=10),
        refvals=pformat({k: {'energy': v[0], 'frame_id': v[1]} for k, v in refvals.items()}),
        missing=missing,
    ), metadata, missing


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


if __name__ == '__main__':
    main()