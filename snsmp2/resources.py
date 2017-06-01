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
