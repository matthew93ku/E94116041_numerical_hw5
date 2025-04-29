import numpy as np
import matplotlib.pyplot as plt

# 定義微分方程系統
def f(t, u):
    u1, u2 = u
    f1 = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    f2 = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([f1, f2])

# 解析解
def exact_solution(t):
    u1 = 2*np.exp(-3*t) - np.exp(-37*t) + (1/3)*np.cos(t)
    u2 = -np.exp(-3*t) + 2*np.exp(-37*t) - (1/3)*np.cos(t)
    return np.array([u1, u2])

# 四階龍格-庫塔方法
def runge_kutta4(t0, u0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    u = np.zeros((len(t), 2))
    u[0] = u0
    for i in range(len(t) - 1):
        k1 = h * f(t[i], u[i])
        k2 = h * f(t[i] + h/2, u[i] + k1/2)
        k3 = h * f(t[i] + h/2, u[i] + k2/2)
        k4 = h * f(t[i] + h, u[i] + k3)
        u[i + 1] = u[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, u

# 參數
t0 = 0.0
u0 = np.array([4/3, 2/3])
t_end = 1.0  # 假設 t 從 0 到 1（題目未明確指定）

# 步長 h = 0.1
h1 = 0.1
t1, u1 = runge_kutta4(t0, u0, t_end, h1)

# 步長 h = 0.05
h2 = 0.05
t2, u2 = runge_kutta4(t0, u0, t_end, h2)

# 計算解析解
u_exact1 = np.array([exact_solution(t) for t in t1])
u_exact2 = np.array([exact_solution(t) for t in t2])

# 比較結果 (h = 0.1)
print("h = 0.1")
print("t\tu1_RK4\tu2_RK4\tu1_Exact\tu2_Exact\tu1_Error\tu2_Error")
for i in range(len(t1)):
    u1_error = abs(u1[i, 0] - u_exact1[i, 0])
    u2_error = abs(u1[i, 1] - u_exact1[i, 1])
    print(f"{t1[i]:.1f}\t{u1[i,0]:.6f}\t{u1[i,1]:.6f}\t{u_exact1[i,0]:.6f}\t{u_exact1[i,1]:.6f}\t{u1_error:.6f}\t{u2_error:.6f}")

# 比較結果 (h = 0.05)
print("\nh = 0.05")
print("t\tu1_RK4\tu2_RK4\tu1_Exact\tu2_Exact\tu1_Error\tu2_Error")
for i in range(len(t2)):
    u1_error = abs(u2[i, 0] - u_exact2[i, 0])
    u2_error = abs(u2[i, 1] - u_exact2[i, 1])
    print(f"{t2[i]:.2f}\t{u2[i,0]:.6f}\t{u2[i,1]:.6f}\t{u_exact2[i,0]:.6f}\t{u_exact2[i,1]:.6f}\t{u1_error:.6f}\t{u2_error:.6f}")

# 繪圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t1, u1[:, 0], 'o-', label='u1 (h=0.1)')
plt.plot(t2, u2[:, 0], 's-', label='u1 (h=0.05)')
plt.plot(t1, u_exact1[:, 0], '-', label='u1 Exact')
plt.xlabel('t')
plt.ylabel('u1')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t1, u1[:, 1], 'o-', label='u2 (h=0.1)')
plt.plot(t2, u2[:, 1], 's-', label='u2 (h=0.05)')
plt.plot(t1, u_exact1[:, 1], '-', label='u2 Exact')
plt.xlabel('t')
plt.ylabel('u2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()