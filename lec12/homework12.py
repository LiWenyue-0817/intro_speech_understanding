import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    
    @param:
    duration (scalar) - length of the excitation, in samples
    F0 (scalar) - pitch frequency, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    excitation (np.ndarray) - the excitation signal, such that
      excitation[n] = -1 if n is an integer multiple of int(np.round(Fs/F0))
      excitation[n] = 0 otherwise
    '''
    # 计算基音周期（样本数）
    pitch_period = int(np.round(Fs / F0))
    # 初始化激励信号为全0
    excitation = np.zeros(duration, dtype=np.float32)
    # 对基音周期的整数倍位置赋值为-1
    for n in range(duration):
        if n % pitch_period == 0:
            excitation[n] = -1
    return excitation

def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    
    @param:
    x (np.ndarray(N)) - the excitation signal
    F (scalar) - resonant frequency, in Hertz
    BW (scalar) - resonant bandwidth, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    y (np.ndarray(N)) - resonant output
    '''
    # 计算数字滤波器参数
    omega = 2 * np.pi * F / Fs  # 谐振角频率
    alpha = np.exp(-np.pi * BW / Fs)  # 衰减系数
    b0 = 1 - alpha**2  # 前向系数
    a1 = -2 * alpha * np.cos(omega)  # 反馈系数1
    a2 = alpha**2  # 反馈系数2
    
    # 初始化输出信号
    y = np.zeros_like(x)
    # 递归计算谐振器输出（二阶IIR滤波器）
    for n in range(2, len(x)):
        y[n] = b0 * x[n] - a1 * y[n-1] - a2 * y[n-2]
    return y

def synthesize_vowel(duration, F0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, Fs):
    '''
    Synthesize a vowel.
    
    @param:
    duration (scalar) - duration in samples
    F0 (scalar) - pitch frequency in Hertz
    F1 (scalar) - first formant frequency in Hertz
    F2 (scalar) - second formant frequency in Hertz
    F3 (scalar) - third formant frequency in Hertz
    F4 (scalar) - fourth formant frequency in Hertz
    BW1 (scalar) - first formant bandwidth in Hertz
    BW2 (scalar) - second formant bandwidth in Hertz
    BW3 (scalar) - third formant bandwidth in Hertz
    BW4 (scalar) - fourth formant bandwidth in Hertz
    Fs (scalar) - sampling frequency in samples/second
    
    @returns:
    speech (np.ndarray(samples)) - synthesized vowel
    '''
    # 1. 生成浊音激励信号
    excitation = voiced_excitation(duration, F0, Fs)
    # 2. 依次通过四个共振峰谐振器（叠加形式）
    res1 = resonator(excitation, F1, BW1, Fs)
    res2 = resonator(excitation, F2, BW2, Fs)
    res3 = resonator(excitation, F3, BW3, Fs)
    res4 = resonator(excitation, F4, BW4, Fs)
    # 3. 叠加所有谐振器输出得到元音信号
    speech = res1 + res2 + res3 + res4
    # 归一化（避免幅值溢出）
    speech = speech / np.max(np.abs(speech))
    return speech

# 测试代码（运行入口）
if __name__ == "__main__":
    # 配置参数（以元音 /a/ 为例，男性发音特征）
    Fs = 16000  # 采样率16kHz
    duration_ms = 500  # 元音时长500ms
    duration = int(duration_ms * Fs / 1000)  # 转换为样本数
    
    # 基频和共振峰参数（/a/ 元音典型值）
    F0 = 100  # 基频100Hz（男性）
    F1 = 730  # 第一共振峰
    F2 = 1090 # 第二共振峰
    F3 = 2440 # 第三共振峰
    F4 = 3010 # 第四共振峰
    # 共振峰带宽（典型值）
    BW1 = 60
    BW2 = 70
    BW3 = 100
    BW4 = 120
    
    # 合成元音
    vowel = synthesize_vowel(duration, F0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, Fs)
    
    # 打印结果信息
    print("合成完成！")
    print(f"元音信号长度：{len(vowel)} 样本")
    print(f"信号幅值范围：{np.min(vowel):.4f} ~ {np.max(vowel):.4f}")
    
    # 可选：绘制波形图（需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        time_axis = np.arange(duration) / Fs  # 时间轴（秒）
        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, vowel)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Synthesized Vowel /a/ Waveform')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("提示：未安装matplotlib，跳过波形绘制。可执行 pip install matplotlib 安装")
