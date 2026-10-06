#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Exercise the built Linux unixlib fd ABI, without opening a GPU device.

Usage: python3 unix-fds.py /absolute/path/to/d3d12.so
"""
import ctypes as c
import errno
import fcntl
import os
import sys


class Dup(c.Structure):
    _fields_ = [('fd', c.c_int32), ('new_fd', c.c_int32),
                ('err', c.c_int32), ('pad', c.c_int32)]


class Size(c.Structure):
    _fields_ = [('fd', c.c_int32), ('err', c.c_int32), ('size', c.c_int64)]


class Close(c.Structure):
    _fields_ = [('fd', c.c_int32), ('pad', c.c_int32)]


class Map(c.Structure):
    _fields_ = [('addr', c.c_uint64), ('length', c.c_uint64),
                ('prot', c.c_int32), ('flags', c.c_int32),
                ('fd', c.c_int32), ('err', c.c_int32), ('offset', c.c_uint64)]


class Unmap(c.Structure):
    _fields_ = [('addr', c.c_uint64), ('length', c.c_uint64),
                ('ret', c.c_int32), ('err', c.c_int32)]


class Open(c.Structure):
    _fields_ = [('fd', c.c_int32), ('err', c.c_int32), ('name', c.c_char * 32)]


lib = c.CDLL(sys.argv[1])
call = c.CFUNCTYPE(c.c_uint32, c.c_void_p)
table = (c.c_void_p * 10).in_dll(lib, '__wine_unix_call_funcs')


def invoke(index, args):
    assert call(table[index])(c.byref(args)) == 0
    return args


fd = os.memfd_create('unixlib-fd-test', os.MFD_CLOEXEC)
# A value outside LLP64 long detects accidental truncation in the PE ABI.
length = 0x100000003
os.ftruncate(fd, length)
size = invoke(9, Size(fd=fd))
assert size.size == length and size.err == 0
dup = invoke(8, Dup(fd=fd))
assert dup.new_fd >= 3 and dup.new_fd != fd and dup.err == 0
assert fcntl.fcntl(dup.new_fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
mapping = invoke(3, Map(length=4096, prot=3, flags=1, fd=dup.new_fd))
assert mapping.addr and mapping.err == 0
c.memmove(mapping.addr, b'unixlib', 7)
assert os.pread(fd, 7, 0) == b'unixlib'
unmap = invoke(4, Unmap(addr=mapping.addr, length=4096))
assert unmap.ret == 0 and unmap.err == 0
invoke(1, Close(fd=fd))
assert os.pread(dup.new_fd, 7, 0) == b'unixlib'
invoke(1, Close(fd=dup.new_fd))
try:
    fcntl.fcntl(dup.new_fd, fcntl.F_GETFD)
except OSError as exc:
    assert exc.errno == errno.EBADF
else:
    raise AssertionError('close leaked the duplicate')
assert invoke(8, Dup(fd=-1)).err == errno.EBADF
assert invoke(9, Size(fd=-1)).size == -1
assert invoke(9, Size(fd=-1)).err == errno.EBADF
# Render-node rejection must close every candidate, including forced nodes.
os.environ['D3D12AGX_RENDER_NODE'] = '/dev/null'
before = set(os.listdir('/proc/self/fd'))
for _ in range(8):
    assert invoke(0, Open()).fd == -1
assert set(os.listdir('/proc/self/fd')) == before
print('UNIX_FDS_RESULT failures=0 (dup CLOEXEC, 64-bit size, mmap, close, rejection)')
