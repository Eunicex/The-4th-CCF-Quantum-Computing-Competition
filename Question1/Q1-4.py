import numpy as np
import pyqpanda as pq
from pyqpanda import QCircuit, CNOT, H, SWAP

# 导入比较矩阵的函数，用于验证结果。禁止修改此处的代码
def compare_matrices(mat1: np.ndarray, mat2: np.ndarray) -> bool:
    """
    Compare two matrices for equality.
    """
    if mat1.shape != mat2.shape:
        return False
    return np.allclose(mat1, mat2, atol=1e-8)


if __name__ == "__main__":
    qvm = pq.CPUQVM()
    qvm.init_qvm()
    qbits = qvm.qAlloc_many(2)

    """
    注意：请在下方开始答题，请勿修改上方的代码!!!
    """