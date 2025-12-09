import numpy as np

def householder_qr(A):
    """
    使用 Householder 变换对矩阵 A 做 QR 分解：A = Q R

    参数：
        A : (m, n) 的二维数组或可转为数组的对象

    返回：
        Q : (m, m) 正交矩阵
        R : (m, n) 上三角矩阵
    """
    A = np.array(A, dtype=float).copy()
    m, n = A.shape
    Q = np.eye(m)

    for k in range(min(m, n)):
        # 取出第 k 列，从第 k 行开始的子向量
        x = A[k:, k]
        if np.allclose(x, 0):
            # 这列下面已经全是 0，跳过
            continue

        # e1 = (1,0,...,0)^T
        e1 = np.zeros_like(x)
        e1[0] = 1.0

        # 构造 Householder 向量 v
        # alpha = -sign(x0) * ||x||
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        # 防止 x[0] = 0 时 sign(0) = 0 导致 alpha = 0
        if alpha == 0:
            alpha = -np.linalg.norm(x)

        u = x - alpha * e1
        v = u / np.linalg.norm(u)

        # H_k = I - 2 v v^T (作用在子块上)
        Hk_sub = np.eye(m - k) - 2.0 * np.outer(v, v)
        Hk = np.eye(m)
        Hk[k:, k:] = Hk_sub

        # 左乘 Hk 消去第 k 列下面的元素
        A = Hk @ A
        # Q = H1^T H2^T ... Hk^T
        Q = Q @ Hk.T

    R = A
    return Q, R


def givens_qr(A):
    """
    使用 Givens 旋转对矩阵 A 做 QR 分解：A = Q R

    参数：
        A : (m, n) 的二维数组或可转为数组的对象

    返回：
        Q : (m, m) 正交矩阵
        R : (m, n) 上三角矩阵
    """
    A = np.array(A, dtype=float).copy()
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        # 逐个消去第 j 列对角线以下元素
        for i in range(m - 1, j, -1):
            if abs(R[i, j]) < 1e-12:
                continue
            a = R[i - 1, j]
            b = R[i, j]

            # 计算 Givens 旋转参数 c, s，使得
            # [ c -s; s c ] [a; b] = [r; 0]
            r = np.hypot(a, b)
            if r == 0:
                continue
            c = a / r
            s = -b / r

            G = np.eye(m)
            G[i - 1, i - 1] = c
            G[i, i] = c
            G[i - 1, i] = -s
            G[i, i - 1] = s

            R = G @ R
            Q = Q @ G.T

    return Q, R


def back_substitute(R, y):
    """
    回代求解上三角线性方程组 R x = y

    参数：
        R : (n, n) 上三角矩阵
        y : (n,) 或 (n,1) 向量

    返回：
        x : (n,) 解向量
    """
    R = np.array(R, dtype=float)
    y = np.array(y, dtype=float).reshape(-1)
    n = R.shape[1]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-12:
            raise ValueError("R 矩阵在对角线处接近奇异，无法回代")
        x[i] = (y[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def solve_qr(A, b, method="householder"):
    """
    利用 QR 分解求解 Ax = b，并在超定/无解情形下给出最小二乘意义下的近似解。

    参数：
        A : (m, n) 系数矩阵
        b : (m,) 或 (m,1) 右端项
        method : 字符串，"householder" 或 "givens"

    返回：
        x        : 近似解（如果有精确解则是精确解），维度 (n,)
        res_norm : 残差范数 ||Ax - b||
        Q, R     : 对应方法得到的 Q, R 矩阵
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[0] != m:
        raise ValueError("b 的维度与 A 不匹配：A 是 %d 行, b 是 %d 行" % (m, b.shape[0]))

    if method.lower() == "householder":
        Q, R = householder_qr(A)
    elif method.lower() == "givens":
        Q, R = givens_qr(A)
    else:
        raise ValueError("未知分解方法：%s，应为 'householder' 或 'givens'" % method)

    # Q^T b
    Qtb = Q.T @ b

    # 区分情形：
    # 1) m >= n: 按常规取前 n 行 R1 做回代，得到最小二乘解
    # 2) m < n: 欠定情形，解的自由度较大，返回最小范数解（minimum-norm solution）
    if m >= n:
        R1 = R[:n, :n]
        y1 = Qtb[:n, :]

        x = np.zeros((n, b.shape[1]))
        for col in range(b.shape[1]):
            x[:, col] = back_substitute(R1, y1[:, col])

        x = x.squeeze()
        residual = A @ x.reshape(-1, 1) - b
        res_norm = np.linalg.norm(residual)

        return x, res_norm, Q, R
    else:
        # 欠定：使用最小范数解 x = A^T (A A^T)^{-1} b，当 A A^T 奇异时退回伪逆
        AAT = A @ A.T
        try:
            invAAT = np.linalg.inv(AAT)
        except np.linalg.LinAlgError:
            invAAT = np.linalg.pinv(AAT)

        x_full = A.T @ (invAAT @ b)
        x = x_full.squeeze()
        residual = A @ x.reshape(-1, 1) - b
        res_norm = np.linalg.norm(residual)

        return x, res_norm, Q, R


def main():
    print("==== 矩阵正交分解与 Ax = b 求解程序 ====")
    print("请选择分解方法：")
    print("  1) Householder reduction")
    print("  2) Givens reduction")
    choice = input("请输入 1 或 2：").strip()

    if choice == "1":
        method = "householder"
    elif choice == "2":
        method = "givens"
    else:
        print("输入错误，默认使用 Householder 方法。")
        method = "householder"

    # 输入矩阵尺寸
    m = int(input("请输入矩阵 A 的行数 m："))
    n = int(input("请输入矩阵 A 的列数 n："))

    print("请按行输入矩阵 A，每行输入 %d 个数，以空格分隔：" % n)
    A = []
    for i in range(m):
        while True:
            row_str = input(f"第 {i+1} 行：").strip()
            try:
                row = [float(x) for x in row_str.split()]
                if len(row) != n:
                    print("该行元素个数不是 %d，请重新输入。" % n)
                    continue
                A.append(row)
                break
            except ValueError:
                print("输入中包含非数字，请重新输入。")

    print("请输入向量 b（长度为 %d），以空格分隔：" % m)
    while True:
        b_str = input("b：").strip()
        try:
            b = [float(x) for x in b_str.split()]
            if len(b) != m:
                print("b 的长度不是 %d，请重新输入。" % m)
                continue
            break
        except ValueError:
            print("输入中包含非数字，请重新输入。")

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # 求解
    x, res_norm, Q, R = solve_qr(A, b, method=method)

    print("\n=== 使用的方法：%s ===" % method)
    print("矩阵 A：")
    print(A)
    print("\n分解得到的 Q（正交矩阵）：")
    print(Q)
    print("\n分解得到的 R（上三角矩阵）：")
    print(R)
    print("\n求得的解 x（最小二乘意义下）：")
    print(x)
    print("\n残差范数 ||Ax - b|| = %.6e" % res_norm)

    # 提示方程是否“近似有解”或“无精确解”
    if res_norm < 1e-8:
        print("判定：残差非常小，可认为方程组有精确解。")
    else:
        print("判定：残差不为零，方程组一般无精确解，上述 x 为最小二乘意义下的近似解。")


if __name__ == "__main__":
    main()