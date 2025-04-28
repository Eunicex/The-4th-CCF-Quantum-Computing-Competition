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
    # 1. 构建CNOT门和H门实现的SWAP电路
    circuit_cnot_h_swap = QCircuit()
    circuit_cnot_h_swap << CNOT(qbits[0], qbits[1])
    circuit_cnot_h_swap << H(qbits[0])
    circuit_cnot_h_swap << H(qbits[1])
    circuit_cnot_h_swap << CNOT(qbits[0], qbits[1])
    circuit_cnot_h_swap << H(qbits[0])
    circuit_cnot_h_swap << H(qbits[1])
    circuit_cnot_h_swap << CNOT(qbits[0], qbits[1])

    # 2. 构建直接使用SWAP门的电路
    circuit_swap = QCircuit()
    circuit_swap << SWAP(qbits[0], qbits[1])

    # 3. 获取两个电路的矩阵表示
    from pyqpanda import *
    matrix_cnot_swap = np.matrix(get_matrix(circuit_cnot_h_swap, False)).reshape((4, 4))
    matrix_swap = np.matrix(get_matrix(circuit_swap, False)).reshape((4, 4))

    # 4. 比较两个矩阵
    are_equal = compare_matrices(matrix_cnot_swap, matrix_swap)

    # 5. 打印比较结果
    print("三个CNOT门构造的SWAP门的矩阵:")
    print(matrix_cnot_swap)
    print("\n普通SWAP门的矩阵:")
    print(matrix_swap)

    if are_equal:
        print("\n得证，三个CNOT构造的电路可实现SWAP功能")

    qvm.finalize()