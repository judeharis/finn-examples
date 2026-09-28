# Jude: Created
"""Vectorised stand-ins for the driver's pack_input / unpack_output.

FINN's finnpy_to_packed_bytearray / packed_bytearray_to_finnpy only have fast paths
for 8-bit and 1-bit datatypes; anything else goes through per-element hex strings.
For this layer (UINT4 in, INT32 out) that is minutes per frame on the KV260's A53s.

Both functions reproduce the driver's convention, reverse_inner=True plus
reverse_endian=True: element 0 of the innermost (SIMD/PE) dimension sits in the
least-significant bits, the packed word is padded up to whole bytes, and bytes are
little-endian. Checked against FINN's own functions by check_fast_packing.py.
"""
import numpy as np


def pack(folded, bits, signed):
    """(..., n) integer-valued array -> (..., ceil(n*bits/8)) uint8, as the driver packs."""
    v = np.asarray(folded).astype(np.int64) & ((1 << bits) - 1)
    n = v.shape[-1]
    nbytes = (n * bits + 7) // 8
    if bits in (8, 16, 32) and (n * bits) % 8 == 0:
        return np.ascontiguousarray(v.astype("<u%d" % (bits // 8))).view(np.uint8)
    # bit-plane expand LSB-first, element 0 first, then repack little-endian
    shifts = np.arange(bits, dtype=np.int64)
    b = ((v[..., None] >> shifts) & 1).astype(np.uint8).reshape(*v.shape[:-1], n * bits)
    pad = nbytes * 8 - n * bits
    if pad:
        b = np.concatenate([b, np.zeros((*b.shape[:-1], pad), dtype=np.uint8)], axis=-1)
    return np.packbits(b, axis=-1, bitorder="little")


def unpack(packed, bits, signed, n):
    """(..., nbytes) uint8 -> (..., n) float32, inverse of pack()."""
    p = np.ascontiguousarray(packed, dtype=np.uint8)
    if bits in (8, 16, 32) and p.shape[-1] * 8 == n * bits:
        dt = ("<i%d" if signed else "<u%d") % (bits // 8)
        return p.view(dt).astype(np.float32)
    b = np.unpackbits(p, axis=-1, bitorder="little")[..., : n * bits]
    b = b.reshape(*p.shape[:-1], n, bits).astype(np.int64)
    v = (b << np.arange(bits, dtype=np.int64)).sum(axis=-1)
    if signed:
        v = np.where(v >= (1 << (bits - 1)), v - (1 << bits), v)
    return v.astype(np.float32)


def install(accel):
    """Swap the overlay's slow pack_input/unpack_output for the vectorised ones."""
    def pack_input(ibuf_folded, ind=0):
        dt = accel.idt(ind)
        return pack(ibuf_folded, dt.bitwidth(), dt.signed())

    def unpack_output(obuf_packed, ind=0):
        dt = accel.odt(ind)
        n = accel.oshape_folded(ind)[-1]
        return unpack(obuf_packed, dt.bitwidth(), dt.signed(), n).reshape(accel.oshape_folded(ind))

    accel.pack_input = pack_input
    accel.unpack_output = unpack_output
