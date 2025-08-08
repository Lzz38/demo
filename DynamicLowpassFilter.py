import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


class DynamicLowpassFilter:
    """
    动态截止频率低通滤波器类
    
    根据信号在不同时间段的频谱特性，动态调整滤波器的截止频率，
    实现对时变信号的智能滤波，保留有用信号，抑制干扰噪声。
    """
    
    def __init__(
        self,
        fs: float,
        segment_duration: float = 0.5,
        energy_ratio: float = 0.9,
        filter_order: int = 4,
        overlap_ratio: float = 0.5,
        smooth_fc_samples: int = 3,
        min_fc: float = 0.1,
        max_fc: float = None,
        use_sos: bool = True,
        spectrum_window: str = 'hann',
    ):
        """
        初始化动态低通滤波器
        
        Args:
            fs: 采样率(Hz) - 决定频率分辨率
            segment_duration: 分段时长（秒）- 时间分辨率与频率分辨率的权衡
            energy_ratio: 能量比例阈值(0~1) - 用于估算截止频率, 如0.9表示保留90%能量
            filter_order: 巴特沃斯滤波器阶数 - 影响滤波器的陡峭程度
            overlap_ratio: 段间重叠比例(0~<1) - 用于overlap-add合成, 减少边界伪影
            smooth_fc_samples: fc平滑窗口长度 - 抑制截止频率的突变
            min_fc: 截止频率下限(Hz) - 防止fc过小导致过度滤波
            max_fc: 截止频率上限(Hz) - 防止fc过大导致滤波不足
            use_sos: 是否使用二阶节滤波 - 提高数值稳定性
            spectrum_window: 频谱估计加窗方式 - 'hann'减少谱泄漏，'none'不加窗
        """
        self.fs = fs
        self.segment_duration = max(float(segment_duration), 1.0 / fs)
        self.energy_ratio = float(energy_ratio)
        self.filter_order = int(filter_order)
        self.overlap_ratio = float(overlap_ratio)
        self.smooth_fc_samples = int(max(1, smooth_fc_samples))
        self.min_fc = float(min_fc) if min_fc is not None else None
        self.max_fc = float(max_fc) if max_fc is not None else None
        self.use_sos = bool(use_sos)
        self.spectrum_window = spectrum_window

    def _compute_segment_and_hop(self, signal_length: int) -> (int, int):
        """
        计算分段长度和跳跃步长
        
        Args:
            signal_length: 信号总长度
            
        Returns:
            segment_length: 每段长度（采样点数）
            hop_length: 段间跳跃步长（采样点数）
        """
        # 根据采样率和期望时长计算段长度
        segment_length = int(round(self.fs * self.segment_duration))
        # 确保段长度在合理范围内
        segment_length = max(1, min(segment_length, signal_length))
        
        # 验证重叠比例的有效性
        if not (0.0 <= self.overlap_ratio < 1.0):
            self.overlap_ratio = 0.0
            
        # 计算跳跃步长：hop = segment * (1 - overlap)
        hop_length = max(1, int(round(segment_length * (1.0 - self.overlap_ratio))))
        return segment_length, hop_length

    def _apply_window(self, seg: np.ndarray) -> np.ndarray:
        """
        对信号段应用窗函数，减少频谱泄漏
        
        Args:
            seg: 输入信号段
            
        Returns:
            加窗后的信号段
        """
        if self.spectrum_window == 'hann' and len(seg) > 1:
            # 汉宁窗：减少频谱泄漏，提高频率估计精度
            return seg * np.hanning(len(seg))
        return seg

    def estimate_cutoff_frequencies(self, signal: np.ndarray):
        """
        分段估算截止频率 - 核心算法
        
        原理：对每个时间段计算功率谱，找到累积能量达到指定比例时的频率点，
        该频率点即为该段的截止频率。这样可以根据信号的时变特性动态调整滤波参数。
        
        Args:
            signal: 输入信号
            
        Returns:
            segment_times: 每段的起始时间（秒）
            estimated_frequencies: 每段估算的截止频率(Hz)
        """
        n = len(signal)
        segment_length, hop_length = self._compute_segment_and_hop(n)
        print(f"Segment length: {segment_length}, Hop length: {hop_length}")
        # 计算每段的起始位置
        starts = np.arange(0, max(1, n - segment_length + 1), hop_length)
        if len(starts) == 0:
            starts = np.array([0])

        estimated_fcs = []
        segment_times = []

        # 逐段分析频谱特性
        for start in starts:
            end = min(start + segment_length, n)
            seg = signal[start:end]  # 提取当前段
            seg_w = self._apply_window(seg)  # 加窗减少谱泄漏

            # 计算功率谱密度
            freqs = np.fft.rfftfreq(len(seg_w), d=1 / self.fs)  # 频率轴
            fft_mag = np.abs(np.fft.rfft(seg_w)) ** 2  # 功率谱
            total_energy = np.sum(fft_mag) + 1e-12  # 总能量（避免除零）
            fft_mag_norm = fft_mag / total_energy  # 归一化功率谱
            
            # 计算累积能量分布
            cumulative_energy = np.cumsum(fft_mag_norm)

            # 找到累积能量达到阈值的位置
            cutoff_idx = np.searchsorted(cumulative_energy, self.energy_ratio)
            cutoff_idx = min(max(cutoff_idx, 0), len(freqs) - 1)  # 边界检查
            fc = float(freqs[cutoff_idx])  # 该频率即为截止频率

            # 应用频率约束
            if self.min_fc is not None:
                fc = max(fc, self.min_fc)  # 下限约束
            if self.max_fc is not None:
                fc = min(fc, self.max_fc)  # 上限约束

            estimated_fcs.append(fc)
            segment_times.append(start / self.fs)  # 转换为时间（秒）

        estimated_fcs = np.asarray(estimated_fcs, dtype=float)
        segment_times = np.asarray(segment_times, dtype=float)

        # 对截止频率序列进行平滑处理，减少突变
        if self.smooth_fc_samples > 1 and len(estimated_fcs) > 1:
            k = self.smooth_fc_samples  # 平滑窗口长度
            pad = k // 2  # 边界填充长度
            
            # 边界填充：镜像填充避免边界效应
            pad_left = estimated_fcs[:pad][::-1] if pad > 0 else np.array([])
            pad_right = estimated_fcs[-pad:][::-1] if pad > 0 else np.array([])
            padded = np.concatenate([pad_left, estimated_fcs, pad_right])
            
            # 滑动平均滤波
            kernel = np.ones(k, dtype=float) / float(k)
            smoothed = np.convolve(padded, kernel, mode='same')
            estimated_fcs = smoothed[pad:pad + len(estimated_fcs)]

        return segment_times, estimated_fcs

    def apply_filter(self, signal: np.ndarray, estimated_fcs: np.ndarray) -> np.ndarray:
        """
        应用动态滤波 - 使用overlap-add技术
        
        原理：对每段信号使用该段估算的截止频率进行滤波，然后通过重叠相加
        和窗函数加权合成最终结果，有效减少段边界处的伪影。
        
        Args:
            signal: 输入信号
            estimated_fcs: 每段估算的截止频率
            
        Returns:
            滤波后的信号
        """
        n = len(signal)
        segment_length, hop_length = self._compute_segment_and_hop(n)
        starts = np.arange(0, max(1, n - segment_length + 1), hop_length)
        if len(starts) == 0:
            starts = np.array([0])

        # 初始化输出和权重数组
        output = np.zeros_like(signal, dtype=float)
        weights = np.zeros_like(signal, dtype=float)

        # 处理fc数量与段数不匹配的情况
        if len(estimated_fcs) != len(starts):
            # 通过线性插值重采样fc序列
            estimated_fcs = np.interp(
                np.arange(len(starts)),
                np.arange(len(estimated_fcs)),
                estimated_fcs,
            )

        # 逐段滤波并合成
        for i, start in enumerate(starts):
            end = min(start + segment_length, n)
            seg = signal[start:end]  # 当前段
            fc = float(estimated_fcs[i])  # 该段的截止频率

            # 计算归一化截止频率（0~1）
            wn = min(max(fc / (self.fs / 2.0), 1e-4), 0.999)
            
            # 设计并应用滤波器
            if self.use_sos:
                # 使用二阶节滤波，提高数值稳定性
                sos = butter(self.filter_order, wn, btype='low', output='sos')
                filtered_seg = sosfiltfilt(sos, seg)  # 零相位滤波
            else:
                # 传统b,a系数滤波
                b, a = butter(self.filter_order, wn, btype='low')
                filtered_seg = filtfilt(b, a, seg)  # 零相位滤波

            # 应用合成窗函数（汉宁窗）
            win = np.hanning(len(seg)) if len(seg) > 1 else np.ones(len(seg))
            output[start:end] += filtered_seg * win  # 加权叠加
            weights[start:end] += win  # 累积权重

        # 归一化处理：消除窗函数的影响
        nonzero = weights > 1e-12
        output[nonzero] /= weights[nonzero]

        # 处理未覆盖的尾部区域
        uncovered = ~nonzero
        if np.any(uncovered):
            output[uncovered] = signal[uncovered]  # 直接复制原始信号

        return output

    def process(self, signal: np.ndarray, visualize: bool = True) -> np.ndarray:
        """
        完整的动态滤波处理流程
        
        Args:
            signal: 输入信号
            visualize: 是否显示可视化结果
            
        Returns:
            滤波后的信号
        """
        # 步骤1：估算每段的截止频率
        _, freqs = self.estimate_cutoff_frequencies(signal)
        
        # 步骤2：应用动态滤波
        filtered_signal = self.apply_filter(signal, freqs)

        # 步骤3：可视化结果（可选）
        if visualize:
            t = np.arange(len(signal)) / self.fs
            plt.figure(figsize=(12, 5))
            plt.plot(t, signal, label='原始信号', alpha=0.5)
            plt.plot(t, filtered_signal, label='动态滤波后', linewidth=2)
            plt.title('动态截止频率低通滤波效果（overlap-add + 平滑fc）')
            plt.xlabel('时间 (秒)')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return filtered_signal


if __name__ == "__main__":

    fs_demo = 1000.0
    duration_demo = 4.0
    t_demo = np.arange(0, duration_demo, 1 / fs_demo)
    signal_demo = np.concatenate([
        0.5 * np.sin(2 * np.pi * 5 * t_demo[:len(t_demo)//2]),
        0.5 * np.sin(2 * np.pi * 5 * t_demo[len(t_demo)//2:]) + 0.3 * np.sin(2 * np.pi * 80 * t_demo[len(t_demo)//2:])
    ])
    # 打印signal_demo的长度和前10个样本
    print(f"Signal length: {len(signal_demo)}, First 10 samples: {signal_demo[:10]}")
    # 使用类进行动态滤波
    dlpf = DynamicLowpassFilter(fs=fs_demo, segment_duration=0.5, energy_ratio=0.9, filter_order=4)
    filtered_signal_demo = dlpf.process(signal_demo)