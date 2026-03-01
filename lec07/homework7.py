import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(X, Fs):
    '''
    Plot the magnitude spectrum of a signal X with sampling frequency Fs.
    
    @param:
    X (array): 1D numpy array of signal samples
    Fs (scalar): sampling frequency in Hz
    
    @result:
    None (plots the spectrum)
    '''
    # 计算FFT并取幅度
    N = len(X)
    X_fft = np.fft.fft(X)
    X_mag = np.abs(X_fft)
    
    # 生成频率轴（0到Fs/2，对应正频率）
    freq = np.fft.fftfreq(N, 1/Fs)
    # 只取正频率部分（避免对称重复）
    positive_freq_idx = freq >= 0
    freq_pos = freq[positive_freq_idx]
    X_mag_pos = X_mag[positive_freq_idx]
    
    # 绘制幅度谱
    plt.figure(figsize=(8, 4))
    plt.plot(freq_pos, X_mag_pos)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    plt.grid(True)
    plt.show()

def make_complex_tone(f, Fs, phase, dur):
    '''
    Generate a complex tone (sum of harmonics) at frequency f with given phase, sampling rate Fs, duration dur.
    
    @param:
    f (scalar): fundamental frequency in Hz
    Fs (scalar): sampling frequency in Hz
    phase (scalar): initial phase in radians
    dur (scalar): duration in seconds
    
    @result:
    x (array): complex tone signal
    '''
    # 生成时间轴
    N = int(Fs * dur)  # 总采样点数
    t = np.arange(N) / Fs
    
    # 生成基频信号（可扩展为谐波和，这里先实现基频）
    # 若需谐波，可循环叠加 2f、3f... 此处先实现核心逻辑
    x = np.cos(2 * np.pi * f * t + phase)
    return x

# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 测试参数
    f0 = 440  # 基频（A4音）
    Fs = 44100  # 采样频率（音频标准）
    phase = np.pi/2  # 初始相位（90度）
    dur = 1  # 时长1秒
    
    # 1. 生成复音信号
    tone_signal = make_complex_tone(f0, Fs, phase, dur)
    print(f"生成的信号长度：{len(tone_signal)} 采样点")
    print(f"信号前5个采样值：{tone_signal[:5]}")
    
    # 2. 绘制信号频谱
    plot_spectrum(tone_signal, Fs)
