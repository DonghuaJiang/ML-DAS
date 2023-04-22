import numpy as np


def RK4(x0, y0, h, n, fs):                                                                # 龙格-库塔算法
    m = len(fs)
    x = np.linspace(x0, x0+(n-1)*h, num=n)
    y = np.zeros(shape=(m, n))
    y[:, 0] = y0
    for i in range(n-1):
        K1 = np.zeros(m)                                                                  # 计算K1
        for index, f in enumerate(fs):
            K1[index] = f(x[i], y[:, i])
        K2 = np.zeros(m)                                                                  # 计算K2
        for index, f in enumerate(fs):
            K2[index] = f(x[i]+h/2, y[:, i]+h*K1/2)
        K3 = np.zeros(m)                                                                  # 计算K3
        for index, f in enumerate(fs):
            K3[index] = f(x[i]+h/2, y[:, i]+h*K2/2)
        K4 = np.zeros(m)                                                                  # 计算k4
        for index, f in enumerate(fs):
            K4[index] = f(x[i]+h, y[:, i]+h*K3)

        y[:, i+1] = y[:, i]+h*(K1+2*K2+2*K3+K4)/6
    return x, y


def map(data, len):                                                                       # 将实数映射至[0,255]范围内的整数
    res = np.zeros(len)
    for i in range(0, len):
        res[i] = np.mod(np.floor(np.multiply(np.absolute(data[i]), np.power(10, 7))), 255)
    return np.uint8(res)


def int_to_bin_matrix(arr: np.ndarray, width: int):                                       # 进制转换函数：+进制->二进制
  return np.array([[int(c) for c in np.binary_repr(i, width=width)] for i in arr], dtype=np.int8)

# Jianxiong Zhang, Wansheng Tang: A novel bounded 4D chaotic system
def NB4D_model(len, key, img_height, img_width):                                          # 非线性有界的四维混沌系统
    a1, a2, a3 = -0.16, -0.35, -0.75
    a4, a5, a6, a7 = -0.15, -0.45, -0.5, -0.4
    b1, b2, b3, b4 = 1.50, 1.10, 1.00, 1.15
    fs = [
        lambda x, y: a1*y[0]+a2*y[3]-y[1]*y[2],
        lambda x, y: -a3*y[0]+a4*y[1]+b1*y[0]*y[2],
        lambda x, y: a5*y[2]+b2*y[0]*y[1]+b3*y[0]*y[3],
        lambda x, y: a6*y[1]+a7*y[3]-b4*y[0]*y[2]
    ]
    x, y = RK4(0, np.array(key), 0.01, len, fs)
    y_ = int_to_bin_matrix(map(y[1], len), 8)
    return y_.reshape(1, img_height, img_width)

# resu = NB4D_model(3, [6.1, -8.2, 3.2, 11.3], 1, 24)
# print(resu)