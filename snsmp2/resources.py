#                       SNS-MP2 LICENSE AGREEMENT
#
# Copyright 2017, D. E. Shaw Research. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#     this list of conditions, and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions, and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
# Neither the name of D. E. Shaw Research nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from __future__ import print_function, division, absolute_import
import re
import os


def vminfo():
    # type: () -> Dict[str, float]
    """Query virtual memory info from /proc/[pid]/status

    VmPeak: Peak virtual memory size.
    VmSize: Virtual memory size.
    VmLck: Locked memory size (see mlock(3)).
    VmHWM: Peak resident set size ("high water mark").
    VmRSS: Resident set size.  Note that the value here is the
    VmData, VmStk, VmExe: Size of data, stack, and text
                          segments.
    VmLib: Shared library code size.
    VmPTE: Page table entries size.
    VmSwap: Swapped-out virtual memory size by anonymous private
                pages; shmem swap usage is not included.

    Returns
    -------
    d : Dict[str, float]
        Different categories of memory usage, in MB.
    """
    fields = ('VmPeak', 'VmSize', 'VmLck', 'VmHWM',
              'VmRSS', 'VmData', 'VmStk', 'VmExe',
              'VmLib', 'VmPTE', 'VmSwap')

    return_value = {}
    with open('/proc/%s/status' % os.getpid()) as f:
        for line in f:
            if any(line.startswith(f) for f in fields):
                m = re.match('(\w+):\s+(\d+)\skB', line)
                return_value[m.group(1)] = int(m.group(2)) / 1024
    return return_value
