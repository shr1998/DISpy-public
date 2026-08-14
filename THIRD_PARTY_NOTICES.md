# Third-Party Notices

DISpy depends on third-party Python packages listed in `requirements.txt`.
Those packages are not included in this repository and remain subject to their
own licenses.

## Surf96 and disba-derived code

The Surf96-derived modules preserve the original notice for *Computer Programs
in Seismology, Volume IV*, copyright 1986 and 1991 by D. R. Russell and R. B.
Herrmann. Computer Programs in Seismology is distributed under the MIT License:

https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/index.html

The Numba/Python Surf96 implementation in `_surf96.py`, `_surf96_np.py`,
`_surf96_vector_gpu.py`, `_surf96_vectorAll_gpu.py`, and `gsurf96.py` is derived
from the BSD-licensed `disba` project:

https://github.com/keurfonluu/disba

BSD 3-Clause License

Copyright (c) 2020, Keurfon Luu

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The DISpy MIT license does not replace or override these upstream terms.

TDMS support uses the separately installed `nptdms` package. This repository
does not include the former Silixa TDMS reader implementation or relicense its
source code.
