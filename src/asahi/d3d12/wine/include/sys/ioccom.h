/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Linux ioctl request encoding for the Wine PE build of d3d12agx. drm.h takes
 * its non-Linux branch under mingw and includes <sys/ioccom.h>; the request
 * numbers still travel verbatim to the Linux kernel through the unixlib, so
 * this must be the asm-generic encoding, not the BSD one.
 */
#ifndef D12_WINE_SYS_IOCCOM_H
#define D12_WINE_SYS_IOCCOM_H

#define _IOC_NRBITS   8
#define _IOC_TYPEBITS 8
#define _IOC_SIZEBITS 14
#define _IOC_DIRBITS  2

#define _IOC_NRSHIFT   0
#define _IOC_TYPESHIFT (_IOC_NRSHIFT + _IOC_NRBITS)
#define _IOC_SIZESHIFT (_IOC_TYPESHIFT + _IOC_TYPEBITS)
#define _IOC_DIRSHIFT  (_IOC_SIZESHIFT + _IOC_SIZEBITS)

#define _IOC_NONE  0U
#define _IOC_WRITE 1U
#define _IOC_READ  2U

#define _IOC(dir, type, nr, size)                                             \
   (((unsigned)(dir) << _IOC_DIRSHIFT) | ((unsigned)(type) << _IOC_TYPESHIFT) | \
    ((unsigned)(nr) << _IOC_NRSHIFT) | ((unsigned)(size) << _IOC_SIZESHIFT))

#define _IO(type, nr)        _IOC(_IOC_NONE, (type), (nr), 0)
#define _IOR(type, nr, st)   _IOC(_IOC_READ, (type), (nr), sizeof(st))
#define _IOW(type, nr, st)   _IOC(_IOC_WRITE, (type), (nr), sizeof(st))
#define _IOWR(type, nr, st)  _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), sizeof(st))

#endif
