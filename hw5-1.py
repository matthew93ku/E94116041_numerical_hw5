import numpy as np
import matplotlib.pyplot as plt

# 定義微分方程 f(t, y)
def f(t, y):
    return 1 + y/t + (y/t)**2

# 解析解
def exact_solution(t):
    return t * np.tan(np.log(t))

# 歐拉方法
def euler_method(t0, y0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return t, y

# 二階泰勒方法的偏導數和 T^(2)
def df_dt(t, y):
    term1 = -y/t**2 - 2*y**2/t**3
    term2 = (1 + y/t + y**2/t**2) * (1/t + 2*y/t**2)
    return term1 + term2

def taylor_order2(t0, y0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t) - 1):
        T2 = f(t[i], y[i]) + (h/2) * df_dt(t[i], y[i])
        y[i + 1] = y[i] + h * T2
    return t, y

# 參數
t0, y0 = 1.0, 0.0
t_end = 2.0
h = 0.1

# 計算數值解
t_euler, y_euler = euler_method(t0, y0, t_end, h)
t_taylor, y_taylor = taylor_order2(t0, y0, t_end, h)

# 計算解析解
y_exact = exact_solution(t_euler)

# 比較結果
print("t\tEuler\tTaylor\tExact\tEuler Error\tTaylor Error")
for i in range(len(t_euler)):
    euler_error = abs(y_euler[i] - y_exact[i])
    taylor_error = abs(y_taylor[i] - y_exact[i])
    print(f"{t_euler[i]:.1f}\t{y_euler[i]:.6f}\t{y_taylor[i]:.6f}\t{y_exact[i]:.6f}\t{euler_error:.6f}\t{taylor_error:.6f}")

# 繪圖
plt.plot(t_euler, y_euler, 'o-', label='Euler')
plt.plot(t_taylor, y_taylor, 's-', label='Taylor Order 2')
plt.plot(t_euler, y_exact, '-', label='Exact')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()