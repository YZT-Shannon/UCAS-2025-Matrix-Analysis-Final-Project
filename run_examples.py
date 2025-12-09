from __future__ import annotations
import os
import sys
import numpy as np

# 导入 qrsolver.py（使用 __file__ 确定脚本目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
import qrsolver


def print_case(name, A, b):
    print("\n--- %s ---" % name)
    print("A =")
    print(A)
    print("b =", b.squeeze())

    for method in ("householder", "givens"):
        x, res_norm, Q, R = qrsolver.solve_qr(A, b, method=method)
        print("\nmethod = %s" % method)
        print("x =", x)
        print("||Ax - b|| = %.6e" % res_norm)
        print("Q @ R (reconstructed) =")
        print(np.round(Q @ R, 10))


if __name__ == '__main__':
    # 方阵（有精确解）
    A1 = np.array([[2.0, 1.0], [1.0, 3.0]])
    b1 = np.array([1.0, 2.0])
    print_case("方阵有解示例", A1, b1)

    # 超定（最小二乘） m > n
    A2 = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b2 = np.array([1.0, 2.0, 2.0])
    print_case("超定最小二乘示例", A2, b2)

    # 欠定 m < n（此处演示回代对上三角子矩阵的行为）
    A3 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b3 = np.array([7.0, 8.0])
    print_case("欠定示例（演示）", A3, b3)

    print('\n示例运行完成。')