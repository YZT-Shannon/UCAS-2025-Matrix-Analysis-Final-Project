# 2025ucas矩阵分析大作业

# QR 分解与求解 Ax = b（Householder 与 Givens 实现）

本项目实现了矩阵正交分解的两种经典方法——**Householder Reflection** 与 **Givens Rotation**，并基于 QR 分解构建了求解线性方程组 (Ax = b) 的统一框架。所有核心算法均按照课堂讲授的数学原理手写实现，**未调用任何现成的 QR 分解库函数**。

项目包含一个综合可执行程序 `qrsolver.py`，可根据用户选择执行不同分解方法，并自动完成 QR 分解、求解方程组、判断是否存在精确解等功能。以及在 `/example` 文件夹下的测试程序 `run_examples.py` 及对应测试结果截图。

---

## 目录

* [实验目的](#实验目的)
* [算法原理概述](#算法原理概述)

  * [Householder 反射](#1-householder-反射)
  * [Givens 旋转](#2-givens-旋转)
  * [QR 分解求解 Ax=b](#3-qr-分解与线性方程组求解)
* [程序结构](#程序结构)
* [主要函数说明](#主要函数说明)
* [综合主程序使用方法](#综合主程序使用方法)
* [示例结果说明](#示例结果说明)
* [实验总结](#实验总结)

---

## 实验目的

1. 理解 Householder 与 Givens 正交变换的数学原理。
2. 掌握正交分解在 QR 分解中的应用。
3. 基于 QR 分解实现线性方程组 (Ax = b) 的求解，包括：

   * 方阵系统的精确解
   * 超定系统的最小二乘解
   * 欠定系统的最小范数解
4. 实现一个综合可执行程序，能够根据用户选择执行不同方法并输出分解结果。

---

## 算法原理概述

### 1. Householder 反射

Householder 反射通过如下形式的反射矩阵对子块进行整体消元：

```math
H = I - 2\,\frac{vv^T}{v^T v}
```

通过选择适当的向量 (v)，一次反射即可将矩阵某列下方全部元素置零。逐列应用反射矩阵即可得到上三角矩阵 (R)。

**优点：高数值稳定性；一次操作消去多个元素。**

---

### 2. Givens 旋转

Givens 旋转通过二维旋转矩阵对矩阵单个元素进行消元：

```math
G =
\begin{bmatrix}
c & -s \\
s & c
\end{bmatrix},
\qquad
r=\sqrt{a^2+b^2},\quad 
c=\frac{a}{r},\quad 
s=-\frac{b}{r}.
```

通过对不同的行对应用旋转矩阵，可逐个消除目标元素。

**优点：适合稀疏矩阵，控制精细。**

---

### 3. QR 分解与线性方程组求解

QR 分解满足：

```math
A = QR,\qquad Q^T Q = I.
```

代入方程：

```math
Ax = b \;\Rightarrow\; Rx = Q^T b.
```

根据矩阵形状不同：

* 若 (A) 满秩方阵：得到精确解
* 若 (m>n)：返回最小二乘解
* 若 (m<n)：返回最小范数解

```math
x = A^T (AA^T)^{-1} b
```
---

## 程序结构

| 文件名                        | 说明                                            |
| -------------------------- | --------------------------------------------- |
| `qrsolver.py`              | 主程序：包含 Householder / Givens / 求解函数与交互式 main() |
| `examples/run_examples.py` | 提供方阵、超定、欠定的示例运行脚本                             |
| `requirements.txt`         | 包含项目依赖（numpy）                                 |

---

## 主要函数说明

### `householder_qr(A)`

**输入：**

* A：形状 (m, n) 的浮点矩阵

**输出：**

* Q：正交矩阵 (m, m)
* R：上三角矩阵 (m, n)

---

### `givens_qr(A)`

与 `householder_qr(A)` 相同，但使用 Givens 旋转实现。

---

### `solve_qr(A, b, method='householder')`

**输入：**

* A：系数矩阵
* b：向量
* method：`'householder'` 或 `'givens'`

**输出：**

* x：解（精确解 / 最小二乘解 / 最小范数解）
* res_norm：残差范数 (|Ax-b|)
* Q, R：QR 分解结果

---

## 综合主程序使用方法

安装依赖：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

运行综合程序：

```bash
python3 qrsolver.py
```

程序将依次提示：

1. 选择分解方法（Householder / Givens）
2. 输入矩阵尺寸 m, n
3. 逐行输入矩阵 A
4. 输入向量 b

程序会输出：

* Q、R
* 解向量 x
* 残差范数 (|Ax-b|)
* 精确解/近似解判断

---

## 示例结果说明

可以运行：

```bash
python3 examples/run_examples.py
```

### 1. 方阵（有精确解）

```math
A =
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix},
\qquad
b =
\begin{bmatrix}
1 \\
2
\end{bmatrix}
```

结果：

```math
x = (0.2,\; 0.6),\qquad \|Ax - b\| = 0.
```

**运行截图：**

<img src="example/square_result.png" width="30%" style="border:1px solid #ccc;">

---

### 2. 超定系统（最小二乘）

```math
A =
\begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix},
\qquad
b = (1,\;2,\;2)
```

结果：

```math
x = (0.6667,\; 0.5), 
\qquad
\|Ax - b\| \approx 0.4082.
```

**运行截图：**

<img src="example/overdetermined_result.png" width="30%" style="border:1px solid #ccc;">

---

### 3. 欠定系统（最小范数解）

程序自动返回满足最小范数条件的解。
**运行截图：**

<img src="example/underdetermined_result.png" width="30%" style="border:1px solid #ccc;">

---

## 实验总结

本项目完整实现了：

* Householder 与 Givens 正交分解算法
* 基于 QR 分解的线性方程组求解
* 对超定与欠定情形的处理
* 一个综合可执行程序
* 自动化示例验证脚本
