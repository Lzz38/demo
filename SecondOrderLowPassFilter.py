import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, lfilter  
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

class SecondOrderLowPassFilter:
    """
    二阶低通滤波器类(基于双线性变换)

    参数:
        fc (float): 数字截止频率(Hz),需小于采样频率的一半
        fs (float): 采样频率(Hz)
        zeta (float): 阻尼比(默认0.707,对应巴特沃斯响应)
    """

    def __init__(self, fc: float, fs: float, zeta: float = 0.707):
        # 参数校验
        if fc >= fs / 2:
            raise ValueError("截止频率必须小于采样频率的一半(奈奎斯特频率)")
        if zeta <= 0:
            raise ValueError("阻尼比必须大于0")

        self.fc = fc  # 数字截止频率
        self.fs = fs  # 采样频率
        self.zeta = zeta  # 阻尼比

        # 步骤1：计算数字截止角频率(用于频率预畸变)
        omega_d = 2 * np.pi * fc / fs  # 数字截止角频率(rad/sample)
        print(omega_d)

        # 步骤2：频率预畸变(双线性变换补偿频率畸变)
        # 计算等效的模拟截止角频率
        omega_n = 2 * fs * np.tan(omega_d / 2)  # 模拟截止角频率(rad/s)
        print(omega_n)

        # 步骤3：计算滤波器系数(基于双线性变换)
        fs_sq = fs ** 2  # 采样频率平方
        omega_n_sq = omega_n ** 2  # 模拟截止角频率平方
        denominator = (4 * fs_sq) + (4 * zeta * omega_n * fs) + omega_n_sq
        print(denominator)

        # 分子系数(对应b0, b1, b2)
        self.b0 = omega_n_sq / denominator
        self.b1 = (2 * omega_n_sq) / denominator
        self.b2 = omega_n_sq / denominator

        # 分母系数(对应a1, a2,a0=1已归一化)
        self.a1 = (-8 * fs_sq + 2 * omega_n_sq) / denominator
        self.a2 = (4 * fs_sq - 4 * zeta * omega_n * fs + omega_n_sq) / denominator

        # 初始化滤波器状态(保存前2个输入和输出样本)
        self.x_prev1 = 0.0  # x[n-1]
        self.x_prev2 = 0.0  # x[n-2]
        self.y_prev1 = 0.0  # y[n-1]
        self.y_prev2 = 0.0  # y[n-2]

    def filter_sample(self, x: float) -> float:
        """
        处理单个输入样本

        参数:
            x (float): 当前输入样本

        返回:
            float: 当前输出样本
        """
        # 计算当前输出(根据差分方程)
        y = (self.b0 * x +
             self.b1 * self.x_prev1 +
             self.b2 * self.x_prev2 -
             self.a1 * self.y_prev1 -
             self.a2 * self.y_prev2)

        # 更新状态变量(滑动窗口)
        self.x_prev2 = self.x_prev1
        self.x_prev1 = x
        self.y_prev2 = self.y_prev1
        self.y_prev1 = y

        return y

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        处理整个信号数组

        参数:
            signal (np.ndarray): 输入信号数组

        返回:
            np.ndarray: 滤波后的信号数组
        """
        return np.array([self.filter_sample(x) for x in signal])



if __name__ == "__main__":
    # 生成测试信号参数
    fs = 1000.0  # 采样频率(Hz)
    duration = 2.0  # 信号时长(秒)
    t = np.arange(0, duration, 1 / fs)  # 时间轴


    # 生成混合信号(低频+高频+噪声)
    signal_clean = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5Hz有用信号
    signal_noise = 0.3 * np.sin(2 * np.pi * 80 * t)  # 80Hz干扰信号
    signal = signal_clean + signal_noise  # 原始含噪信号
    # 设计滤波器(截止频率10Hz,保留5Hz信号,滤除80Hz干扰)
    fo = SecondOrderLowPassFilter(fc=10.0, fs=fs, zeta=0.707)

    # 应用滤波器
    signal_filtered = fo.filter_signal(signal)

    # --------------------------- 结果可视化 ---------------------------
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, signal, 'b-', alpha=0.6, label='含噪信号')
    plt.plot(t, signal_clean, 'g--', linewidth=2, label='5Hz有用信号')
    plt.title('原始信号(含5Hz有用信号和80Hz干扰)')
    plt.ylabel('幅值')
    plt.xlabel('时间 (s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1 / fs)
    fft_original = np.abs(np.fft.rfft(signal)) / n
    fft_filtered = np.abs(np.fft.rfft(signal_filtered)) / n
    plt.semilogy(freq, fft_original, 'b-', alpha=0.6, label='原始信号')
    plt.semilogy(freq, fft_filtered, 'g--', linewidth=2, label='滤波后信号')
    plt.title('频域响应(对数刻度)')
    plt.ylabel('幅值(归一化)')
    plt.xlabel('频率 (Hz)')
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    # 滤波后时域信号
    plt.subplot(3, 1, 3)
    plt.plot(t, signal_filtered, 'r-', linewidth=2, label='滤波后信号')
    plt.plot(t, signal_clean, 'g--', linewidth=2, label='5Hz有用信号')
    plt.title('滤波后信号')
    plt.ylabel('幅值')
    plt.xlabel('时间 (s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --------------------------- 与scipy库函数对比 ---------------------------
    # 使用scipy的butter函数设计二阶低通滤波器
    b_scipy, a_scipy = butter(N=2, Wn=10, btype='low', fs=fs, analog=False)

    signal_filtered_scipy = lfilter(b_scipy, a_scipy, signal)

    print("手动计算系数 vs scipy库系数对比:")
    print(f"b0: {fo.b0:.6f} vs {b_scipy[0]:.6f}")
    print(f"b1: {fo.b1:.6f} vs {b_scipy[1]:.6f}")
    print(f"b2: {fo.b2:.6f} vs {b_scipy[2]:.6f}")
    print(f"a1: {fo.a1:.6f} vs {a_scipy[1]:.6f}")  
    print(f"a2: {fo.a2:.6f} vs {a_scipy[2]:.6f}")  