Spin-network-scaled MP2 (SNS-MP2)
=================================

This module implements the SNS-MP2 method for computing dimer interaction
energies described by McGibbon et al. [1]. It is implemented as a plugin
for the Psi4 electronic structure method, and requires Psi4 version 1.1
or greater.

Installation
------------
- First, you need to install a working copy of Psi4 1.1 or greater. Head to
  [their website](http://www.psicode.org/psi4manual/master/build_obtaining.html)
  for installation instructions.
- Next, install this plugin using the following commands
```
# Grab the path to the Python interpreter used by your copy of Psi4
$ PSI4_PYTHON=$(head $(which psi4) -n 1 | sed -r 's/^.{2}//')

# Install the SNS-MP2 package with this copy of Python.
$ PSI4_PYTHON -m pip install .
```

Running calculations
--------------------

Here's probably the simplest possible input file. It computes the
interaction energy between two helium atoms separated by two angstroms.

```
molecule {
He 0 0 0
--
He 2 0 0

}

import snsmp2
energy('sns-mp2')
```


Copy the contents to a file called `first-cak.dat.`. To run the calculation,
execute

```
$ psi4 first-calc.dat
```

After it finishes, you can find the results in `first-calc.out`.




References
----------
[1]  R. T. McGibbon, A. G. Taube, A. G. Donchev, K. Siva, F. Fernandez, C. Hargus,
      K.-H. Law, J.L. Klepeis, and D. E. Shaw. "Improving the accuracy of
      Moller-Plesset perturbation theory with neural networks"