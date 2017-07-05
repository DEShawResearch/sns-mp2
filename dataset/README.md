Data Set
--------

The file `interaction_energies.csv.bz2` is a BZIP2-compressed comma-separated
values file containing the electronic structure data used to fit the SNS-MP2
model.

Fields
------

The data file contains 27 fields, which are described here. The field names
each begin with 'm_', 'x_', or 'y_'. Field names beginning with 'm_' denote
metadata fields, those beginning with 'x_' denote input data fields, and those
beginning with 'y_' denoute output/target fields. All united fields are in
atomic units unless otherwise noted.

Metadata fields:

- m_system_name [string]
    
    Example: '{O1COCOC1}_{c1ncc[nH]1}'

    Name of the chemical system. Each chemical system is a dimer interaction
    energy. The `m_system_name` field is formed from the SMILES string of the
    two molecules, wrapped in braces, and separated with an  underscore.

- m_system_molecule_class [string]

    Example: '{ethers}_{pyrrole}'

    Each molecule was assigned to a "molecule class" based on the presence of
    different common organic functional groups. This field, in the same format
    as `m_system_name`, shows the molecule class for each of the two molecules
    in the chemical system.

- m_scan_id [int]

    Example: 1316463

    The data points corresponding to a given chemical system are organized into
    "scans", each of which contains one or more data points. The data points
    in a scan correspond to geometries in which one of two molecules was
    displaced along the intermolecular displacement vector, varrying the
    distance while keeping the relative orientation of the molecules fixed.

    This field is a unique identifier for the scan.

- m_scan_radial_offset [int]

    Example: 1

    This field indicates the position of the frame within the scan. The value
    is the difference between the intermolecular distance of the frame and the
    intermolecular distance of the MP2 energy minima along the displacement
    axis: negative values correspond to compact, repulsive conformations;
    positive values correpnd to more distant, attractive conformations. The
    values are reported in units of 0.1 Angstroms.

- m_ccsdt_delta_basis [str]

    Example: desavtz

    Basis set in which the counterpoise-corrected CCSD(T) was performed for
    computing the reference interaction energy. The basis sets are described
    in the appendix of the paper. The basis set is one of {"desavdz",
    "desavtz", "desavtz(d/p)", "desavqz"}, which correspond to {"AVDZ", "AVTZ",
    "AVTZ(d/p)" and "AVQZ"} in the notation used in the paper.

- m_mp2_intene_lo_basis [str]

    Example: "desavtz"

    Basis set in which the low-basis counterpoise-corrected MP2 calculation
    was performed for computing both the reference interaction energy as
    well as the low-basis interaction energy features.


Data fields

- x_hf_lo_energy [float]

    Small-basis counterpoise-corrected HF interaction energy. This value was
    computed in the basis set indicated by `m_mp2_intene_lo_basis`.

- x_hf_qz_energy [float]

    Large-basis counterpoise-corrected HF interaction energy. This value was
    computed in the AVQZ basis set.

- x_mp2_lo_energy [float]

    Small-basis counterpoise-corrected MP2 interaction energy. This value was
    computed in the basis set indicated by `m_mp2_intene_lo_basis`.

- x_mp2_lo_sscorl_energy [float]

    Same-spin component of the counterpoise-corrected MP2 interaction
    correlation energy. This value was computed in the basis set indicated by
    `m_mp2_intene_lo_basis`.

- x_mp2_qz_energy [float]

    Large-basis counterpoise-corrected MP2 interaction energy. This value was
    computed in the AVQZ basis set.

- x_mp2_qz_sscorl_energy [float]

    Same-spin component of the counterpoise-corrected MP2 interaction
    correlation energy. This value was computed in the AVQZ basis set.

- x_hf_qz_es_energy [float]

    Hartree-fock electrostatic interaction energy. This value was computed in
    the AVQZ basis set.

- x_hf_qz_hl_energy [float]

    Heitler-London interaction energy. This value was computed in the AVQZ basis
    set.

- x_hf_qz_ovl [float]

    Density matrix overlap of the two fragments, computed from their HF monomer
    density matricies in the AVQZ basis set.

- x_mp2_qz_ovl [float]

    Density matrix overlap of the two fragments, computed from their MP2 monomer
    density matricies in the AVQZ basis set.

- x_sapt_tz_disp20_energy [float]
    
    SAPT0 second-order dispersion energy, computed in the AVTZ basis set.

- x_sapt_tz_disp20_ss_energy [float]

    SAPT0 same-spin second-order dispersion energy, computed in the AVTZ basis
    set.

- x_sapt_tz_elst10r_energy [float]

    SAPT0 first-order electrostatic energy, computed in the AVTZ basis set.

- x_sapt_tz_exch10_energy [float]

    SAPT0 first-order exchange energy, computed in the AVTZ basis set

- x_sapt_tz_exch10s2_energy [float]

    SAPT0 first-order exchange energy (S^2 approximation), computed in the
    AVTZ basis set

- x_sapt_tz_exchdisp20_energy [float]
    
    SAPT0 second-order exchange-dispersion energy, computed in the AVTZ basis
    set

- x_sapt_tz_exchdisp20_ss_energy [float]

    SAPT0 same-spin second-order exchange-dispersion energy, computed in the
    AVTZ basis set

- x_sapt_tz_exchind20r_energy [float]

    SAPT0 second-order exchange-induction energy, computed in the AVTZ basis
    set

- x_sapt_tz_ind20r_energy [float]

    SAPT0 second-order induction energy, computed in the AVTZ basis set

- y_ccsdt_cbs_energy [float]

    Reference interaction energy. This value is computed from a Helgaker-
    extrapolated MP2/CBS interaction energy followed by the addition the
    difference between the CCSD(T) and MP2 interaction energy computed in the 
    `m_ccsdt_delta_basis` basis.