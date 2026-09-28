# Jude: Created
"""Host-side check: fast_packing must match FINN's own driver packing byte for byte.

    python3 board/check_fast_packing.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FINN_DIR = os.environ.get("FINN_DIR", "/mnt/Crucial/WorkspaceB/AMD/finn")
sys.path.insert(0, os.path.join(FINN_DIR, "src"))
from qonnx.core.datatype import DataType  # noqa: E402
from qonnx.util.basic import gen_finn_dt_tensor  # noqa: E402

from finn.util.data_packing import finnpy_to_packed_bytearray, packed_bytearray_to_finnpy  # noqa: E402
import fast_packing  # noqa: E402

np.random.seed(0)
ok = True
cases = [(DataType[d], n) for d in ("UINT4", "INT4", "UINT8", "INT8", "INT21", "INT32") for n in (1, 2, 3, 4, 8)]
for dt, n in cases:
    x = gen_finn_dt_tensor(dt, (2, 5, 7, 3, n))
    ref = finnpy_to_packed_bytearray(x, dt, reverse_inner=True, reverse_endian=True, fast_mode=True)
    got = fast_packing.pack(x, dt.bitwidth(), dt.signed())
    back_ref = packed_bytearray_to_finnpy(ref, dt, x.shape, reverse_inner=True, reverse_endian=True, fast_mode=True)
    back = fast_packing.unpack(got, dt.bitwidth(), dt.signed(), n).reshape(x.shape)
    good = ref.shape == got.shape and np.array_equal(ref, got) and np.array_equal(back, back_ref) \
        and np.array_equal(back, x)
    ok &= good
    print(f"{dt.name:6s} n={n}: packed {tuple(got.shape)} {'OK' if good else 'MISMATCH'}")
print("ALL OK" if ok else "FAILURES")
sys.exit(0 if ok else 1)
