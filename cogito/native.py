
import ctypes
import os
import sys
import numpy as np

# Load the DLL
dll_name = "cogito.dll" if os.name == 'nt' else "libcogito.so"
dll_path = None

# Search Order: Env Var -> Bundled -> Build Dir
# System paths handled by fallback

search_paths = []

if env_path := os.environ.get("COGITO_LIBRARY_PATH"):
    search_paths.append(env_path)

search_paths.append(os.path.dirname(__file__))
search_paths.append(os.path.join(os.path.dirname(__file__), "../../../cogito/build/Debug"))

# 1. Exact path search
for p in search_paths:
    if p:
        candidate = os.path.join(p, dll_name)
        if os.path.exists(candidate):
            dll_path = candidate
            break

# 2. System Fallback
if not dll_path:
    dll_path = dll_name

try:
    _lib = ctypes.CDLL(dll_path)
except OSError as e:
    raise RuntimeError(f"Could not load {dll_name}. Checked paths: {search_paths}. Error: {e}") from e

# --- C Types ---

class CG_Tensor(ctypes.Structure):
    pass

# We only need the pointer type for most operations
CG_Tensor_p = ctypes.POINTER(CG_Tensor)

# --- Function Signatures ---

# cg_tensor* cg_tensor_new(int* shape, int ndim, bool requires_grad);
_lib.cg_tensor_new.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool]
_lib.cg_tensor_new.restype = CG_Tensor_p

# cg_tensor* cg_tensor_from_data(float* data, int* shape, int ndim, bool requires_grad);
_lib.cg_tensor_from_data.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool]
_lib.cg_tensor_from_data.restype = CG_Tensor_p

# void cg_tensor_free(cg_tensor* t);
_lib.cg_tensor_free.argtypes = [CG_Tensor_p]
_lib.cg_tensor_free.restype = None

# void cg_tensor_add(cg_tensor* a, cg_tensor* b, cg_tensor* out);
_lib.cg_tensor_add.argtypes = [CG_Tensor_p, CG_Tensor_p, CG_Tensor_p]
_lib.cg_tensor_add.restype = None

# void cg_tensor_mul(cg_tensor* a, cg_tensor* b, cg_tensor* out);
_lib.cg_tensor_mul.argtypes = [CG_Tensor_p, CG_Tensor_p, CG_Tensor_p]
_lib.cg_tensor_mul.restype = None

class CG_Tensor_Full(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("grad", ctypes.POINTER(ctypes.c_float)),
        ("shape", ctypes.c_int * 8),     # CG_MAX_DIMS = 8
        ("strides", ctypes.c_int * 8),   # CG_MAX_DIMS = 8
        ("ndim", ctypes.c_int),
        ("size", ctypes.c_int),
        ("requires_grad", ctypes.c_bool),
    ]

# Cast return to Full struct pointer when we need field access
def as_full(t_p):
    return ctypes.cast(t_p, ctypes.POINTER(CG_Tensor_Full))

