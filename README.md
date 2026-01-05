
# libcogito

Python bindings for the [Cogito](https://github.com/cogito-ml/cogito) machine learning library.

## Features

- **Zero-Copy NumPy Integration**: Views Cogito tensors as NumPy arrays without copying memory (`__array_interface__`).
- **ctypes bindings**: Lightweight, "close-to-the-metal" integration.
- **Efficient**: Minimal python overhead.

## Installation

```bash
pip install .
```

Or for development:
```bash
pip install -e .
```

## Configuration

If the `cogito` shared library (`cogito.dll` or `libcogito.so`) is not bundled or in system paths, set the environment variable:

```bash
# Windows (PowerShell)
$env:COGITO_LIBRARY_PATH = "C:\path\to\cogito\build\Debug"

# Linux/Mac
export COGITO_LIBRARY_PATH="/path/to/cogito/build"
```

## Usage

```python
import numpy as np
from cogito.tensor import Tensor

# Create a tensor from NumPy data (Zero Copy if contiguous)
data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t = Tensor(data)

# Perform operations
result = t * t

# View result as NumPy array (Zero Copy)
result_view = np.array(result, copy=False)
print(result_view)
# [[ 1.  4.]
#  [ 9. 16.]]
```
