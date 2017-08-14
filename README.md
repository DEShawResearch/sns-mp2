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

License
-------

                      SNS-MP2 LICENSE AGREEMENT

Copyright 2017, D. E. Shaw Research. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

    * Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions, and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

Neither the name of D. E. Shaw Research nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The file `snsmp2/contextdecorator.py` is copyright Michael Foord and is
redistributed under the 3-clause BSD license (see `nsmp2/contextdecorator.py`
for details).

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
