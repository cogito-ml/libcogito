
import sys
import os
import numpy as np

# ensure we can import cogito
sys.path.append(os.getcwd())

from cogito.tensor import Tensor

def test_bindings():
    print("Testing LibCogito Bindings...")
    
    # 1. Create from Shape
    t1 = Tensor([2, 3])
    print(f"Created tensor from shape: {t1.shape}")
    assert t1.shape == (2, 3)
    
    # 2. Create from Data
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    t2 = Tensor(data)
    print(f"Created tensor from data: {t2.shape}")
    
    # 3. Verify Zero-Copy (Reading)
    print("Verifying Zero-Copy Read...")
    view = np.array(t2, copy=False)
    print(f"View:\n{view}")
    assert np.allclose(view, data)
    assert view.ctypes.data == np.array(t2, copy=False).ctypes.data # Should be same pointer
    
    # 4. Math
    print("Testing Math (Add)...")
    t3 = t2 + t2
    view3 = np.array(t3, copy=False)
    print(f"Result:\n{view3}")
    expected = data + data
    assert np.allclose(view3, expected)
    
    print("PASS: All binding tests passed.")

if __name__ == "__main__":
    test_bindings()
