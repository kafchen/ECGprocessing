import numpy as np
import math
import scipy.signal as sig
from Filters import LMS_AdaFilter, RLS_AdaFilter, noise_generator,getBestmu
import matplotlib.pyplot as plt

# 计算信噪比 算滤波前 原始信号/加噪信号 和 滤波后的 原始信号/滤波信号
# 公式: snr = 10lg(Ps/Pn) 单位: dB


def getSNR(s, sn):
    """
    :param s: 纯信号 List[float]
    :param sn: 加噪信号or滤波信号 List[float]
    :return: snr 信噪比 单位:dB
    """
    Ps = np.sum((s) ** 2)/ (len(s))
    Pn = np.sum((sn - s) ** 2)/ (len(sn))
    SNR = 10 * math.log((Ps / Pn), 10)
    return SNR


# 计算MSE 算滤波前 原始信号/加噪信号 和 滤波后的 原始信号/滤波信号
def getMSE(s, sn):
    """
    :param s: 纯信号 List[float]
    :param sn: 加噪信号or滤波信号 List[float]
    :return: 均方误差
    """
    return sum((s - sn) ** 2) / len(s)


def getFeature(data, f=128):
    """
    :param data: 滤波后信号数据
    :param f: 采样频率 即1s有多少个数据点
    :return RR_period, HR: RR间期 （s) 和 心率（次/分）
    """
    data_mean = np.mean(data)
    # p_id 为定位的R波序号值
    p_id, _ = sig.find_peaks(data, distance=60, height=data_mean)

    delta = np.array(p_id[1:]) - np.array(p_id[:-1])
    m = np.mean(delta)
    delta = list(filter(lambda x: 0.9 * m <= x <= 1.2 * m, delta))
    RR_period = np.mean(np.array(delta)) / f
    HR = 60 / RR_period

    return RR_period, HR


if __name__ == "__main__":
    ECG_data = np.load('./ECG-data_npy/ECG1.npy')[:-10]
    reference_data = noise_generator(ECG_data, 10)
    noise_data = ECG_data + reference_data
    # plt.figure()
    # plt.plot(reference_data[:2000])
    # plt.show()
    filter_data_LMS = LMS_AdaFilter(reference_data, noise_data)
    filter_data_RLS = RLS_AdaFilter(reference_data, noise_data)
    #
    # plt.figure()
    # plt.plot(noise_data[:2000])
    # plt.show()
    #

    # bm = getBestmu(noise_data)
    # print(bm)

    # plt.figure()
    # plt.plot(ECG_data[-1000:])
    # plt.show()
    # plt.figure()
    # plt.plot(filter_data_LMS[-1000:])
    # plt.show()


    print("\n\n评价指标")
    # 评价指标——信噪比
    NSR_Noise = getSNR(ECG_data, noise_data)
    NSR_LMS = getSNR(ECG_data[:len(filter_data_RLS)], filter_data_LMS)
    NSR_RLS = getSNR(ECG_data[:len(filter_data_RLS)], filter_data_RLS)

    print(f"原始信噪比 {NSR_Noise}\n"
          f"LMS算法过滤后信噪比 {NSR_LMS}\n"
          f"RLS算法过滤后信噪比 {NSR_RLS}")

    print("\n")
    # 评价指标——均方差
    MSE_Noise = getMSE(ECG_data, noise_data)
    MSE_LMS = getMSE(ECG_data[:len(filter_data_RLS)], filter_data_LMS)
    MSE_RLS = getMSE(ECG_data[:len(filter_data_RLS)], filter_data_RLS)

    print("原始误差 MSE= %.6f\n"
          "LMS算法过滤后 MSE= %.6f\n"
          "RLS算法过滤后 MSE= %.6f" % (MSE_Noise, MSE_LMS, MSE_RLS))

    print("\n")
    rr_, hr = getFeature(filter_data_LMS)
    print(f"LMS算法滤波后心电信号特征：\n"
          f"\t心率： {round(hr, 3)}/分钟\n"
          f"\t心电周期：{round(rr_, 5)}毫米")

    print("\n")
    rr_, hr = getFeature(filter_data_RLS)
    print(f"RLS算法滤波后心电信号特征：\n"
          f"\t心率： {round(hr, 3)}/分钟\n"
          f"\t心电周期：{round(rr_, 5)}毫米")
