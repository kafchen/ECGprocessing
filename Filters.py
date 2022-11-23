import random

import numpy as np
from numpy.fft import rfft as fft
from numpy.fft import irfft as ifft


def LMS_AdaFilter(r, x, N=4, mu_=0.05):
    """
    LMS 自适应滤波器
    :param r: 参考信号，与纯信号无关，与噪声信号相关
    :param x: 加噪信号
    :param N: 窗口大小
    :param mu_: 学习率
    :return: 过滤后信号
    """
    L = min(len(r), len(x))
    h = np.zeros(N)
    e = np.zeros(L - N)
    for n_ in range(L - N):
        r_n = r[n_:n_ + N][::-1]
        x_n = x[n_]
        y_n = np.dot(h, r_n.T)
        e_n = x_n - y_n
        h = h + mu_ * e_n * r_n
        e[n_] = e_n
    return e


def NLMS_AdaFilter(r, x, N=4, mu_=1):
    """
    N-LMS 自适应滤波器
    :param r: 参考信号
    :param x: 加噪信号
    :param N: 窗口大小
    :param mu_: 学习率
    :return: 过滤后信号
    """
    L = min(len(r), len(x))
    h = np.zeros(N)
    e = np.zeros(L - N)
    for n_ in range(L - N):
        r_n = r[n_:n_ + N][::-1]
        x_n = x[n_]
        y_n = np.dot(h, r_n.T)
        e_n = x_n - y_n
        h = h + mu_ * e_n * r_n / (np.dot(r_n, r_n) + 1e-8)
        e[n_] = e_n
    return e


def RLS_AdaFilter(r, x, N=4, alpha=0.999, delta=0.0002):
    """
    RLS 自适应滤波器
    :param r: 参考信号
    :param x: 加噪信号
    :param N: 窗口大小
    :param alpha:遗忘因子
    :param delta:
    :return: 过滤后信号
    """
    L = min(len(r), len(x))
    alpha_inv = 1 / alpha
    h = np.zeros((N, 1))
    P = np.eye(N) / delta
    e = np.zeros(L - N)
    for n in range(L - N):
        r_n = np.array(r[n:n + N][::-1]).reshape(N, 1)
        x_n = x[n]
        y_n = np.dot(r_n.T, h)
        e_n = x_n - y_n
        g = np.dot(P, r_n)
        g = g / (alpha + np.dot(r_n.T, g))
        h = h + e_n * g
        P = alpha_inv * (P - np.dot(g, np.dot(r_n.T, P)))
        e[n] = e_n
    return e


def Kalman_AdaFilter(x, d, N=64, beta=0.9, sgm2u=1e-2, sgm2v=1e-6):
    """
    卡尔曼自适应滤波器
    :param x: 参考信号
    :param d: 加噪信号
    :param N: 窗口大小
    :param beta:
    :param sgm2u:
    :param sgm2v:
    :return: 过滤后信号
    """
    L = min(len(x), len(d))
    Q = np.eye(N) * sgm2v
    R = np.array([sgm2u]).reshape(1, 1)
    H = np.zeros((N, 1))
    P = np.eye(N) * sgm2v
    I = np.eye(N)

    e = np.zeros(L - N)
    for n in range(L - N):
        x_n = np.array(x[n:n + N][::-1]).reshape(1, N)
        d_n = d[n]
        y_n = np.dot(x_n, H)
        e_n = d_n - y_n
        R = beta * R + (1 - beta) * (e_n ** 2)
        Pn = P + Q
        K = np.dot(Pn, x_n.T) / (np.dot(x_n, np.dot(Pn, x_n.T)) + R)
        H = H + np.dot(K, e_n)
        P = np.dot(I - np.dot(K, x_n), Pn)
        e[n] = e_n
    return e


def Freq_Domain_NLMS_AdaFilter(x, d, M, mu=0.05, beta=0.9):
    """
    频域下的 N-LMS 自适应滤波器
    :param x: 参考信号
    :param d: 加噪信号
    :param M: 窗口大小
    :param mu: 学习率
    :param beta:
    :return: 过滤后信号
    """
    H = np.zeros(M + 1, dtype=np.complex)
    norm = np.full(M + 1, 1e-8)

    window = np.hanning(M)
    x_old = np.zeros(M)

    num_block = len(x) // M
    e = np.zeros(num_block * M)

    for n in range(num_block):
        x_n = np.concatenate([x_old, x[n * M:(n + 1) * M]])
        d_n = d[n * M:(n + 1) * M]
        x_old = x[n * M:(n + 1) * M]

        X_n = np.fft.rfft(x_n)
        y_n = ifft(H * X_n)[M:]
        e_n = d_n - y_n

        e_fft = np.concatenate([np.zeros(M), e_n * window])
        E_n = fft(e_fft)

        norm = beta * norm + (1 - beta) * np.abs(X_n) ** 2
        G = mu * E_n / norm
        H = H + X_n.conj() * G

        h = ifft(H)
        h[M:] = 0
        H = fft(h)

        e[n * M:(n + 1) * M] = e_n

    return e


def Freq_Domain_Kalman_AdaFilter(x, d, M, beta=0.95, sgm2u=1e-2, sgm2v=1e-6):
    """
    频域下的卡尔曼自适应滤波器
    :param x: 参考信号
    :param d: 加噪信号
    :param M: 窗口大小
    :param beta:
    :param sgm2u:
    :param sgm2v:
    :return: 过滤后信号
    """
    Q = sgm2u
    R = np.full(M + 1, sgm2v)
    H = np.zeros(M + 1, dtype=np.complex)
    P = np.full(M + 1, sgm2u)

    window = np.hanning(M)
    x_old = np.zeros(M)

    num_block = len(x) // M
    e = np.zeros(num_block * M)

    for n in range(num_block):
        x_n = np.concatenate([x_old, x[n * M:(n + 1) * M]])
        d_n = d[n * M:(n + 1) * M]
        x_old = x[n * M:(n + 1) * M]

        X_n = np.fft.rfft(x_n)

        y_n = ifft(H * X_n)[M:]
        e_n = d_n - y_n

        e_fft = np.concatenate([np.zeros(M), e_n * window])
        E_n = fft(e_fft)

        R = beta * R + (1.0 - beta) * (np.abs(E_n) ** 2)
        P_n = P + Q * (np.abs(H))
        K = P_n * X_n.conj() / (X_n * P_n * X_n.conj() + R)
        P = (1.0 - K * X_n) * P_n

        H = H + K * E_n
        h = ifft(H)
        h[M:] = 0
        H = fft(h)

        e[n * M:(n + 1) * M] = e_n

    return e


# 信号加噪
def noise_generator(x, snr):
    """
    信号噪声生成器
    :param x: 原始信号
    :param snr: 信噪比
    :return: 生成的噪声
    """
    snr = 10 ** (snr / 10.0)
    x_power = np.sum(np.abs(x) ** 2) / len(x)
    n_power = x_power / snr
    k = len(x) / 10
    nos = []
    for i in range(int(k)):
        random_u = np.random.rand(1)
        random_sig = np.random.rand(1)
        nos += list(np.random.normal(random_u, random_sig, 10) * np.sqrt(n_power))

    return np.array(nos)


def getBestmu(x):
    return 2/sum(x**2)


import matplotlib.pyplot as plt

if __name__ == '__main__':
    # f0 = 0.05
    # n = 200
    # t = np.arange(n)
    xs = np.load('./ECG-data_npy/ECG1.npy')
    xs = xs[:-10]
    ws = noise_generator(xs, 10)

    # M = 20
    # xn = 0.05*np.random.randn(xs.shape[0])
    xn = ws + xs
    dn = ws
    en = RLS_AdaFilter(xn, dn)
    plt.figure(1)
    plt.plot(xs[-1500:])
    plt.title("primary data")
    # plt.savefig("org.png", bbox_inches='tight')
    plt.show()

    plt.figure(2)
    plt.plot(dn[-2000:])
    plt.title("noise data")
    plt.show()

    plt.figure(3)
    plt.plot(en[-2000:])
    plt.title("filter data")
    plt.show()
