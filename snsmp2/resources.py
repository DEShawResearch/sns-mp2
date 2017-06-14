# Copyright 2017  D. E. Shaw Research, LLC
# 
# All rights reserved.
# 
# D. E. Shaw Research, LLC ("DESRES") hereby grants you a limited,
# revocable license to use and/or modify the attached computer code (the
# "Code") for internal purposes only. Redistribution of any kind and in
# any form is strictly prohibited.
# 
# In consideration of the rights granted to you hereunder, you hereby
# agree to indemnify, defend and hold DESRES harmless from and against
# any and all damages sustained by DESRES, which damages arise out of or
# relate to your use of the Code, including without limitation any
# damages caused by any allegation or claim that the Code infringes the
# rights of any third party.
# 
# THE ATTACHED CODE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL DESRES BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS CODE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


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
