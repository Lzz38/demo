# coding=utf-8

import os
import re
import platform

'''
The statvfs module defines constants so interpreting the result if os.statvfs(),
which returns a tuple, can be made without remembering “magic numbers.”
Each of the constants defined in this module is the index of the
entry in the tuple returned by os.statvfs() that contains the specified information.

statvfs.F_BSIZE
Preferred file system block size.

statvfs.F_FRSIZE
Fundamental file system block size.

statvfs.F_BLOCKS
Total number of blocks in the filesystem.

statvfs.F_BFREE
Total number of free blocks.

statvfs.F_BAVAIL
Free blocks available to non-super user.

statvfs.F_FILES
Total number of file nodes.

statvfs.F_FFREE
Total number of free file nodes.

statvfs.F_FAVAIL
Free nodes available to non-super user.

statvfs.F_FLAG
Flags. System dependent: see statvfs() man page.

statvfs.F_NAMEMAX
Maximum file name length.
'''


def disk_stat():
    sys = platform.system().lower()
    if sys == 'darwin':
        root_dir = '/'
    elif sys == 'linux':
        root_dir = '/home'
    else:
        return '100%', 0

    disk = os.statvfs(root_dir)
    return '{}%'.format(round((1.0 - float(disk.f_bfree) / disk.f_blocks) * 100, 2)), disk.f_frsize * disk.f_blocks


def uptime_stat():
    try:
        uptime_out = os.popen('uptime').read()
        return re.search('\s*(.*)\s*$', uptime_out).groups()[0]
    except:
        return 'unknown'


if __name__ == '__main__':
    print(disk_stat())
    print(uptime_stat())
