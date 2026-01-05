
import ctypes
import numpy as np
from .native import _lib, CG_Tensor_p, as_full

class Tensor:
    def __init__(self, data_or_shape, requires_grad=False):
        """
        Create a new Tensor.
        
        Args:
            data_or_shape: Either a list/tuple for shape (creates zeros), or a NumPy array/list for data.
            requires_grad: Whether to track gradients.
        """
        self._ptr = None # Underlying CG_Tensor_p
        
        if isinstance(data_or_shape, (list, tuple)) and all(isinstance(x, int) for x in data_or_shape):
            # Create from shape (zeros)
            shape = data_or_shape
            ndim = len(shape)
            shape_array = (ctypes.c_int * ndim)(*shape)
            # Create uninitialized tensor
            self._ptr = _lib.cg_tensor_new(shape_array, ndim, requires_grad)
            
            # TODO: Initialize with zeros if desired, currently uninitialized memory
            
            # Fill with zeros manually or expose cg_tensor_zeros?
            
        elif isinstance(data_or_shape, (np.ndarray, list)):
            # Create from data
            data = np.array(data_or_shape, dtype=np.float32) # Ensure float32
            
            # Check contiguous
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            
            # Create from data
            data = np.array(data_or_shape, dtype=np.float32) # Ensure float32
            
            # Check contiguous
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
           
            # Create tensor from existing data (copies by default in C)
            
            shape = data.shape
            ndim = len(shape)
            shape_array = (ctypes.c_int * ndim)(*shape)
            data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            self._ptr = _lib.cg_tensor_from_data(data_ptr, shape_array, ndim, requires_grad)
        else:
            raise ValueError("Invalid argument for Tensor")
            
        if not self._ptr:
            raise RuntimeError("Failed to allocate tensor")

    def __del__(self):
        if self._ptr:
            _lib.cg_tensor_free(self._ptr)

    @property
    def shape(self):
        full = as_full(self._ptr).contents
        return tuple(full.shape[i] for i in range(full.ndim))

    @property
    def __array_interface__(self):
        """
        Zero-copy NumPy Interface.
        Allows `np.array(tensor, copy=False)` to view underlying memory.
        """
        full = as_full(self._ptr).contents
        shape = tuple(full.shape[i] for i in range(full.ndim))
        # data ptr
        data_ptr = ctypes.addressof(full.data.contents) if full.data else 0
        
        return {
            'shape': shape,
            'typestr': '<f4', # Little-endian float32
            'data': (data_ptr, False), # (ptr, read_only)
            'version': 3
        }

    def __repr__(self):
        return f"Tensor({np.array(self, copy=False)})"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operands must be Tensors")
        # We need to know shape of output. Assume same shape for now.
        out = Tensor(self.shape) 
        _lib.cg_tensor_add(self._ptr, other._ptr, out._ptr)
        return out

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Operands must be Tensors")
        out = Tensor(self.shape)
        _lib.cg_tensor_mul(self._ptr, other._ptr, out._ptr)
        return out
